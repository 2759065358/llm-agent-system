from datetime import datetime
from typing import List
from hello_agents.tools import Tool
from hello_agents.memory import MemoryConfig
from memory.memory_manager import MemoryManager
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
import os
class MemoryTool(Tool):
    """记忆工具 - 为Agent提供记忆功能"""

    def __init__(
        self,
        user_id: str = "default_user",
        memory_config: MemoryConfig = None,
        memory_types: List[str] = None
    ):
        super().__init__(
            name="memory",
            description="记忆工具 - 可以存储和检索对话历史、知识和经验"
        )

        self.memory_config = memory_config or MemoryConfig()
        self.memory_types = memory_types or ["working", "episodic", "semantic"]

        # 使用自己实现的 MemoryManager
        self.memory_manager = MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            memory_types=self.memory_types
        )

        self.current_session_id = None

    def execute(self, action: str, **kwargs) -> str:
        """执行记忆操作"""

        if action == "add":
            return self._add_memory(**kwargs)

        elif action == "search":
            return self._search_memory(**kwargs)

        elif action == "forget":
            return self._forget(**kwargs)

        elif action == "consolidate":
            return self._consolidate(**kwargs)

        else:
            return f" 未知操作: {action}"

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        **metadata
    ) -> str:

        try:

            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            metadata.update({
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            })

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

        except Exception as e:
            return f" 添加记忆失败: {str(e)}"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_types: List[str] = None,
        memory_type: str = None,
        min_importance: float = 0.0,  
        **kwargs                       
    ) -> str:

        try:

            if memory_type and not memory_types:
                memory_types = [memory_type]

            results = self.memory_manager.retrieve_memories(
                query=query,
                limit=limit,
                memory_types=memory_types
            )

            # ✅ 加这一段过滤
            if min_importance > 0:
                results = [m for m in results if m.importance >= min_importance]

            if not results:
                return f"🔍 未找到与 '{query}' 相关的记忆"

            output = [f"🔍 找到 {len(results)} 条相关记忆:"]

            for i, m in enumerate(results, 1):

                preview = m.content[:80]

                output.append(
                    f"{i}. [{m.memory_type}] {preview} (重要性: {m.importance:.2f})"
                )

            return "\n".join(output)

        except Exception as e:
            return f" 搜索记忆失败: {str(e)}"

    def _forget(
        self,
        threshold: float = 0.1
    ) -> str:

        try:

            count = self.memory_manager.forget_memories(
                threshold=threshold
            )

            return f"🧹 已遗忘 {count} 条记忆"

        except Exception as e:
            return f" 遗忘记忆失败: {str(e)}"

    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> str:

        try:

            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold
            )

            return f"🔄 已整合 {count} 条记忆（{from_type} → {to_type}）"

        except Exception as e:
            return f" 整合记忆失败: {str(e)}"
    def run(self, input: Dict[str, Any]) -> str:
        """
        标准Tool入口（框架调用这个）
        """
        action = input.get("action")

        if not action:
            return "❌ 缺少 action 参数"

        kwargs = {k: v for k, v in input.items() if k != "action"}

        return self.execute(action, **kwargs)


    def get_parameters(self) -> Dict[str, Any]:
        """
        告诉Agent这个工具怎么用
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "search", "forget", "consolidate"],
                    "description": "要执行的操作"
                },
                "query": {
                    "type": "string",
                    "description": "搜索查询（用于 search）"
                },
                "content": {
                    "type": "string",
                    "description": "记忆内容（用于 add）"
                },
                "memory_type": {
                    "type": "string",
                    "description": "记忆类型"
                }
            },
            "required": ["action"]
        }