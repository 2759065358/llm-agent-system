from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter
from hello_agents.tools.response import ToolResponse
from hello_agents.tools.error_codes import ToolErrorCode
from rag.rag_pipeline import SimpleRAGPipeline


class RAGTool(Tool):

    def __init__(self):
        super().__init__(
            name="rag",
            description="RAG知识库工具"
        )

        self.pipeline = SimpleRAGPipeline()

    def run(self, input: Dict[str, Any]) -> ToolResponse:

        action = input.get("action")
        if not action:
            action = "search"

        if action == "search":
            query = input.get("query", "")
            top_k = int(input.get("top_k", 5))
            result = self.pipeline.retrieve(query, top_k=top_k)
            return ToolResponse.success(
                text="\n".join(result) if result else "未检索到相关内容",
                data={"chunks": result, "top_k": top_k}
            )

        elif action == "add":
            content = input.get("content", "")
            self.pipeline.add_document(content)
            return ToolResponse.success(text="✅ 已加入知识库", data={"content": content})

        return ToolResponse.error(
            code=ToolErrorCode.INVALID_PARAMETER,
            message=f"❌ 未知action: {action}"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="动作类型：search 或 add",
                required=False,
                default="search"
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索查询（action=search 时使用）",
                required=False
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回条数（action=search 时使用）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="content",
                type="string",
                description="待写入知识库内容（action=add 时使用）",
                required=False
            ),
        ]
