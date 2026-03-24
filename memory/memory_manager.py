from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from dotenv import load_dotenv
load_dotenv()
from hello_agents.memory import (
    MemoryItem,
    MemoryConfig,
    WorkingMemory,
    EpisodicMemory,
)



class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口（轻量版）"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        memory_types: Optional[List[str]] = None
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id

        memory_types = memory_types or ["working", "episodic"]

        self.memory_types: Dict[str, Any] = {}

        if "working" in memory_types:
            self.memory_types["working"] = WorkingMemory(self.config)

        if "episodic" in memory_types:
            self.memory_types["episodic"] = EpisodicMemory(self.config)


    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = False
    ):
        """添加记忆"""

        if memory_type not in self.memory_types:
            raise ValueError(f"不支持的记忆类型: {memory_type}")

        # 简单默认重要性
        if importance is None:
            importance = 0.5

        memory_module = self.memory_types[memory_type]

        memory = MemoryItem(
            id=str(uuid.uuid4()),
            user_id=self.user_id,              
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            timestamp=datetime.now()           
        )


        return memory_module.add(memory)

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0
    ):
        """检索记忆"""

        results = []
        search_types = memory_types or list(self.memory_types.keys())

        # 避免某个类型垄断
        per_type_limit = max(1, limit // len(search_types))

        for mtype in search_types:
            memory_module = self.memory_types.get(mtype)

            if memory_module:
                try:
                    memories = memory_module.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id
                    )
                    results.extend(memories)
                except Exception:
                    continue

        
        results.sort(key=lambda x: getattr(x, "importance", 0), reverse=True)

        return results[:limit]

    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ):
        """遗忘机制"""

        removed = 0

        for memory_module in self.memory_types.values():
            if hasattr(memory_module, "forget"):
                removed += memory_module.forget(
                    strategy=strategy,
                    threshold=threshold,
                    max_age_days=max_age_days
                )

        return removed

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ):
        """记忆整合（迁移高重要性记忆）"""

        from_memory = self.memory_types.get(from_type)
        to_memory = self.memory_types.get(to_type)

        if not from_memory or not to_memory:
            return 0

        # 假设 memory 内部有 get_all
        all_memories = getattr(from_memory, "get_all", lambda: [])()

        candidates = [
            m for m in all_memories
            if getattr(m, "importance", 0) >= importance_threshold
        ]

        count = 0

        for memory in candidates:
            # 先删除再添加（避免重复）
            if hasattr(from_memory, "remove"):
                from_memory.remove(memory.id)

            memory.memory_type = to_type
            memory.importance = min(1.0, memory.importance * 1.1)

            to_memory.add(memory)
            count += 1

        return count