from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import re


class BaseReranker(ABC):
    """rerank 抽象接口，便于后续切换 Cross-Encoder。"""

    @abstractmethod
    def rerank(self, query: str, docs: List[str], top_k: int) -> List[str]:
        raise NotImplementedError


class SimpleKeywordReranker(BaseReranker):
    """简单 rerank：基于 query/doc 关键词重叠计分。"""

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(t) > 1]

    def _score(self, query: str, doc: str) -> float:
        q = set(self._tokenize(query))
        d = set(self._tokenize(doc))

        if not q or not d:
            return 0.0

        overlap = len(q & d)
        return overlap / len(q)

    def rerank(self, query: str, docs: List[str], top_k: int) -> List[str]:
        scored: List[Tuple[float, str]] = [(self._score(query, d), d) for d in docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]


class CrossEncoderReranker(BaseReranker):
    """
    预留 Cross-Encoder 接口。

    TODO:
    - 接入 sentence-transformers CrossEncoder 或在线 rerank API。
    - 在 rerank(query, docs, top_k) 内返回按相关度排序后的 docs。
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def rerank(self, query: str, docs: List[str], top_k: int) -> List[str]:
        # 占位实现：先保持原顺序，避免影响当前稳定性。
        return docs[:top_k]
