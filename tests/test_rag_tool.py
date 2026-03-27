import unittest

from rag.rag_tool import RAGTool
from hello_agents.tools.response import ToolResponse


class _DummyPipeline:
    def retrieve(self, query, top_k=5):
        return [
            {
                "id": "p1",
                "score": 0.91,
                "rank": 1,
                "content": "这是检索内容",
                "source": "/tmp/doc.md",
                "chunk_index": 0,
            }
        ]

    def add_document(self, content):
        return None


class TestRAGTool(unittest.TestCase):
    def _build_tool(self):
        # 避免执行 RAGTool.__init__（它会初始化真实向量库/embedding）
        tool = RAGTool.__new__(RAGTool)
        tool.name = "rag"
        tool.description = "RAG知识库工具"
        tool.pipeline = _DummyPipeline()
        return tool

    def test_search_returns_structured_metadata(self):
        tool = self._build_tool()
        resp = RAGTool.run(tool, {"action": "search", "query": "测试", "top_k": 1})

        self.assertIsInstance(resp, ToolResponse)
        self.assertIn("results", resp.data)
        self.assertEqual(resp.data["results"][0]["source"], "/tmp/doc.md")
        self.assertIn("score=", resp.text)


if __name__ == "__main__":
    unittest.main()
