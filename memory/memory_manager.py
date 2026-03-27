from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid


@dataclass
class MemoryConfig:
    """轻量内存记忆配置。"""
    decay_factor: float = 0.98


@dataclass
class MemoryItem:
    id: str
    user_id: str
    content: str
    memory_type: str
    importance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryManager:
    """记忆管理器 - 纯本地轻量实现（不依赖 hello_agents.memory）"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        memory_types: Optional[List[str]] = None
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.enabled_types = set(memory_types or ["working", "episodic"])
        self._store: Dict[str, List[MemoryItem]] = {
            memory_type: [] for memory_type in self.enabled_types
        }

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = False
    ) -> str:
        if memory_type not in self._store:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        score = 0.5 if importance is None else max(0.0, min(1.0, float(importance)))
        item = MemoryItem(
            id=str(uuid.uuid4()),
            user_id=self.user_id,
            content=str(content),
            memory_type=memory_type,
            importance=score,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        self._store[memory_type].append(item)
        return item.id

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        query_lower = (query or "").lower()
        search_types = memory_types or list(self._store.keys())

        results: List[MemoryItem] = []
        for mtype in search_types:
            for item in self._store.get(mtype, []):
                if item.importance < min_importance:
                    continue
                content_lower = (item.content or "").lower()
                if not query_lower or query_lower in content_lower:
                    results.append(item)

        results.sort(
            key=lambda m: (m.importance, m.timestamp.timestamp()),
            reverse=True
        )
        return results[:limit]

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        removed = 0
        cutoff = datetime.now() - timedelta(days=max_age_days)

        for mtype, items in self._store.items():
            keep: List[MemoryItem] = []
            for item in items:
                too_old = item.timestamp < cutoff
                low_importance = item.importance < threshold
                should_remove = (strategy == "age_based" and too_old) or (
                    strategy != "age_based" and low_importance
                )
                if should_remove:
                    removed += 1
                else:
                    keep.append(item)
            self._store[mtype] = keep

        return removed

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> int:
        if from_type not in self._store or to_type not in self._store:
            return 0

        source = self._store[from_type]
        target = self._store[to_type]

        keep: List[MemoryItem] = []
        moved = 0
        for item in source:
            if item.importance >= importance_threshold:
                item.memory_type = to_type
                item.importance = min(1.0, item.importance * 1.1)
                target.append(item)
                moved += 1
            else:
                keep.append(item)

        self._store[from_type] = keep
        return moved
