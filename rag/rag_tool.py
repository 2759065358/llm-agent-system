from typing import Dict, Any
from hello_agents.tools import Tool
from rag.rag_pipeline import SimpleRAGPipeline


class RAGTool(Tool):

    def __init__(self):
        super().__init__(
            name="rag",
            description="RAG知识库工具"
        )

        self.pipeline = SimpleRAGPipeline()

    def run(self, input: Dict[str, Any]) -> str:

        action = input.get("action")

        if action == "search":
            query = input.get("query", "")
            result = self.pipeline.retrieve(query)
            return str(result)

        elif action == "add":
            content = input.get("content", "")
            self.pipeline.add_document(content)
            return "✅ 已加入知识库"

        return "❌ 未知action"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "query": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["action"]
        }