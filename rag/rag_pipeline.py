from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class SimpleRAGPipeline:

    def __init__(self):

        self.collection_name = os.getenv("QDRANT_COLLECTION", "rag_collection")

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        self.embed_type = os.getenv("EMBED_MODEL_TYPE", "local")

        if self.embed_type == "local":
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(
                os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
            )
            self.vector_size = 384
        else:
            self.embedder = None
            self.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", 1024))
            self.embed_api_key = os.getenv("EMBED_API_KEY")
            self.embed_base_url = os.getenv("EMBED_BASE_URL")

        self._init_collection()

    # =========================
    # 初始化集合
    # =========================
    def _init_collection(self):

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

    # =========================
    # 文档加载入口
    # =========================
    def _load_file(self, path: str) -> str:

        if path.endswith(".txt") or path.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        elif path.endswith(".epub"):
            return self._read_epub(path)

        elif path.endswith(".pdf"):
            return self._read_pdf(path)

        else:
            raise ValueError(f"不支持的文件类型: {path}")

    # ===== EPUB =====
    def _read_epub(self, path: str) -> str:
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(path)
        texts = []

        for item in book.get_items():
            if item.get_type() == 9:
                soup = BeautifulSoup(item.get_content(), "html.parser")

                # ✅ 去掉script/style
                for tag in soup(["script", "style"]):
                    tag.extract()

                text = soup.get_text(separator="\n")

                # ✅ 清理空行
                lines = [line.strip() for line in text.splitlines()]
                lines = [line for line in lines if line]

                texts.append("\n".join(lines))

        return "\n\n".join(texts)

    # ===== PDF =====
    def _read_pdf(self, path: str) -> str:
        from pypdf import PdfReader

        reader = PdfReader(path)
        texts = []

        for page in reader.pages:
            texts.append(page.extract_text() or "")

        return "\n\n".join(texts)

    # =========================
    # Embedding
    # =========================
    def _embed(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        # ===== 本地模型 =====
        if self.embed_type == "local":
            vectors = self.embedder.encode(texts)
            return [
                v.tolist() if hasattr(v, "tolist") else v
                for v in vectors
            ]

        # ===== DashScope（分批处理）=====
        url = f"{self.embed_base_url}/embeddings"

        all_vectors = []
        batch_size = 10  # ⚠️ 官方限制

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


    def add_document(self, content: str):

        # 👉 自动判断是文件还是文本
        if os.path.exists(content):
            print(f"📄 解析文件: {content}")
            text = self._load_file(content)
        else:
            text = content

        chunks = self._split_text(text)
        print(f"✂️ chunk数量: {len(chunks)}")

        vectors = self._embed(chunks)

        points = []

        for i, chunk in enumerate(chunks):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vectors[i],
                "payload": {"content": chunk}
            })

        print(f"🚀 写入向量数量: {len(points)}")

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print("✅ 写入完成")

    # =========================
    # 检索
    # =========================
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:

        query_vector = self._embed(query)[0]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [r.payload["content"] for r in results]

    # =========================
    # chunk
    # =========================
    def _split_text(self, text: str) -> List[str]:

        max_len = 400
        overlap = 80

        import re
        sentences = re.split(r'(。|！|？|\.)', text)

        chunks = []
        current = ""

        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + sentences[i+1]

            if len(current) + len(sentence) < max_len:
                current += sentence
            else:
                chunks.append(current.strip())
                current = current[-overlap:] + sentence

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 50]