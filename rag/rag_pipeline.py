from __future__ import annotations

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid
import os
import requests
import re
from dotenv import load_dotenv

from rag.reranker import BaseReranker, SimpleKeywordReranker

load_dotenv()


class SimpleRAGPipeline:

    CHUNK_SIZE_MAP = {
        "100": 100,
        "300": 300,
        "500": 500,
    }

    def __init__(
        self,
        collection_name: Optional[str] = None,
        chunk_strategy: Optional[str] = None,
        reranker: Optional[BaseReranker] = None,
    ):

        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "rag_collection")

        self.client = None
        self.use_local_store = False
        self.local_points: List[Dict[str, Any]] = []
        try:
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
        except Exception as e:
            print(f"⚠️ Qdrant 初始化失败，切换本地检索模式: {e}")
            self.use_local_store = True

        self.embed_type = os.getenv("EMBED_MODEL_TYPE", "local")
        self.use_simple_embedder = False

        if self.embed_type == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(
                    os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
                )
            except Exception as e:
                print(f"⚠️ sentence-transformers 不可用，切换简易embedding: {e}")
                self.embedder = None
                self.use_simple_embedder = True
            self.vector_size = 384
        else:
            self.embedder = None
            self.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", 1024))
            self.embed_api_key = os.getenv("EMBED_API_KEY")
            self.embed_base_url = os.getenv("EMBED_BASE_URL")

        self.chunk_strategy = str(chunk_strategy or os.getenv("RAG_CHUNK_STRATEGY", "300"))
        self.chunk_size = self.CHUNK_SIZE_MAP.get(self.chunk_strategy, 300)
        self.overlap = max(20, int(self.chunk_size * 0.2))

        self.reranker = reranker or SimpleKeywordReranker()

        self._init_collection()

    def _init_collection(self):

        if self.use_local_store:
            return

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"⚠️ Qdrant 集合初始化失败，切换本地检索模式: {e}")
            self.use_local_store = True

    def _load_file(self, path: str) -> str:

        if path.endswith(".txt") or path.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        if path.endswith(".epub"):
            return self._read_epub(path)

        if path.endswith(".pdf"):
            return self._read_pdf(path)

        raise ValueError(f"不支持的文件类型: {path}")

    def _read_epub(self, path: str) -> str:
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(path)
        texts = []

        for item in book.get_items():
            if item.get_type() == 9:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                for tag in soup(["script", "style"]):
                    tag.extract()

                text = soup.get_text(separator="\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                texts.append("\n".join(lines))

        return "\n\n".join(texts)

    def _read_pdf(self, path: str) -> str:
        from pypdf import PdfReader

        reader = PdfReader(path)
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(texts)

    def _embed(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        if self.embed_type == "local" and not self.use_simple_embedder:
            vectors = self.embedder.encode(texts)
            return [v.tolist() if hasattr(v, "tolist") else v for v in vectors]
        if self.embed_type == "local" and self.use_simple_embedder:
            return [self._simple_embed(t) for t in texts]

        url = f"{self.embed_base_url}/embeddings"
        all_vectors = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.embed_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": os.getenv("EMBED_MODEL_NAME"),
                    "input": batch
                }
            )

            data = resp.json()

            if "data" in data:
                vectors = [item["embedding"] for item in data["data"]]
            elif "output" in data and "embeddings" in data["output"]:
                vectors = [item["embedding"] for item in data["output"]["embeddings"]]
            else:
                raise ValueError(f"Embedding接口异常: {data}")

            all_vectors.extend(vectors)

        return all_vectors

    def _simple_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.vector_size
        tokens = re.findall(r"[\w\u4e00-\u9fff]", text.lower())
        for t in tokens:
            idx = ord(t[0]) % self.vector_size
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _cosine(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        return sum(a * b for a, b in zip(v1, v2))

    def add_document(self, content: str):

        if os.path.exists(content):
            print(f"📄 解析文件: {content}")
            text = self._load_file(content)
        else:
            text = content

        chunks = self._split_text(text)
        print(f"✂️ chunk策略={self.chunk_strategy}, chunk数量={len(chunks)}")

        vectors = self._embed(chunks)

        points = []
        for i, chunk in enumerate(chunks):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vectors[i],
                "payload": {
                    "content": chunk,
                    "chunk_strategy": self.chunk_strategy,
                    "chunk_index": i,
                }
            })

        if self.use_local_store:
            self.local_points.extend(points)
        else:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        print("✅ 写入完成")

    def retrieve(self, query: str, top_k: int = 5, enable_rerank: bool = True) -> List[str]:

        query_vector = self._embed(query)[0]

        candidate_k = max(top_k * 3, top_k)
        if self.use_local_store:
            scored = []
            for p in self.local_points:
                score = self._cosine(query_vector, p.get("vector", []))
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            docs = [p.get("payload", {}).get("content", "") for _, p in scored[:candidate_k]]
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=candidate_k
            )
            docs = [r.payload.get("content", "") for r in results]
        docs = [d for d in docs if d]

        if enable_rerank and self.reranker:
            return self.reranker.rerank(query=query, docs=docs, top_k=top_k)

        return docs[:top_k]

    def _split_text(self, text: str) -> List[str]:
        sentences = re.split(r'(。|！|？|\.|\n)', text)

        chunks = []
        current = ""

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + sentences[i + 1]

            if len(current) + len(sentence) <= self.chunk_size:
                current += sentence
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = current[-self.overlap:] + sentence

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 20]
