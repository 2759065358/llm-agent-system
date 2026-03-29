from typing import Dict, Any
from hello_agents.tools import Tool
from rag.rag_pipeline import SimpleRAGPipeline


class RAGTool(Tool):

    def __init__(self):
        super().__init__(
            name="rag",
            description="RAG知识库工具（支持chunk策略与rerank）"
        )

        self.pipeline = SimpleRAGPipeline()

    def run(self, input: Dict[str, Any]) -> str:

        action = input.get("action")

        if action == "search":
            query = input.get("query", "")
            top_k = int(input.get("top_k", 5))
            enable_rerank = bool(input.get("enable_rerank", True))
            result = self.pipeline.retrieve(
                query=query,
                top_k=top_k,
                enable_rerank=enable_rerank,
            )
            return str(result)

        if action == "add":
            content = input.get("content", "")
            self.pipeline.add_document(content)
            return f"✅ 已加入知识库（chunk策略={self.pipeline.chunk_strategy}）"

        return "❌ 未知action"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "query": {"type": "string"},
                "content": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "enable_rerank": {"type": "boolean", "default": True},
            },
            "required": ["action"]
        }
