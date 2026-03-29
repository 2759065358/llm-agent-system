from fastapi import FastAPI
from pydantic import BaseModel

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.registry import ToolRegistry
from memory.memory_tool import MemoryTool
from rag.rag_tool import RAGTool
from agent.agent import CodeAgent

# ===== 初始化 =====
app = FastAPI()

llm = HelloAgentsLLM()
tool_registry = ToolRegistry()

tool_registry.register_tool(MemoryTool())
tool_registry.register_tool(RAGTool())

agent = CodeAgent(llm, tool_registry)


# ===== 请求结构 =====
class QueryRequest(BaseModel):
    query: str


# ===== 接口 =====
@app.post("/chat")
def chat(req: QueryRequest):
    result = agent.run(req.query)

    # 统一返回字符串，便于前端直接展示
    answer = result.get("final") if isinstance(result, dict) else str(result)
    return {"answer": answer}
