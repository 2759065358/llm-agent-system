from datetime import datetime
from typing import List
from hello_agents.tools import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.error_codes import ToolErrorCode
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

    def execute(self, action: str, **kwargs) -> ToolResponse:
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
            return ToolResponse.error(
                code=ToolErrorCode.INVALID_PARAMETER,
                message=f"未知操作: {action}"
            )

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        **metadata
    ) -> ToolResponse:

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

            return ToolResponse.success(
                text=f"✅ 记忆已添加 (ID: {memory_id[:8]}...)",
                data={"memory_id": memory_id}
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"添加记忆失败: {str(e)}"
            )

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_types: List[str] = None,
        memory_type: str = None,
        min_importance: float = 0.0,  
        **kwargs                       
    ) -> ToolResponse:

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
                return ToolResponse.success(
                    text=f"🔍 未找到与 '{query}' 相关的记忆",
                    data={"results": []}
                )

            output = [f"🔍 找到 {len(results)} 条相关记忆:"]

            for i, m in enumerate(results, 1):

                preview = m.content[:80]

                output.append(
                    f"{i}. [{m.memory_type}] {preview} (重要性: {m.importance:.2f})"
                )

            return ToolResponse.success(
                text="\n".join(output),
                data={
                    "results": [
                        {
                            "memory_type": m.memory_type,
                            "content": m.content,
                            "importance": m.importance
                        } for m in results
                    ]
                }
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"搜索记忆失败: {str(e)}"
            )

    def _forget(
        self,
        threshold: float = 0.1
    ) -> ToolResponse:

        try:

            count = self.memory_manager.forget_memories(
                threshold=threshold
            )

            return ToolResponse.success(
                text=f"🧹 已遗忘 {count} 条记忆",
                data={"forgotten": count}
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"遗忘记忆失败: {str(e)}"
            )

    def _consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> ToolResponse:

        try:

            count = self.memory_manager.consolidate_memories(
                from_type=from_type,
                to_type=to_type,
                importance_threshold=importance_threshold
            )

            return ToolResponse.success(
                text=f"🔄 已整合 {count} 条记忆（{from_type} → {to_type}）",
                data={"consolidated": count}
            )

        except Exception as e:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"整合记忆失败: {str(e)}"
            )

    def run(self, input: Dict[str, Any]) -> ToolResponse:
        """
        标准Tool入口（框架调用这个）
        """
        action = input.get("action")

        if not action:
            return ToolResponse.error(
                code=ToolErrorCode.MISSING_PARAMETER,
                message="缺少 action 参数"
            )

        kwargs = {k: v for k, v in input.items() if k != "action"}

        return self.execute(action, **kwargs)


    def get_parameters(self) -> List[ToolParameter]:
        """
        告诉Agent这个工具怎么用
        """
        return [
            ToolParameter(
                name="action",
                type="string",
                description="要执行的操作：add/search/forget/consolidate",
                required=True
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询（用于 search）",
                required=False
            ),
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容（用于 add）",
                required=False
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型",
                required=False,
                default="working"
            ),
        ]
