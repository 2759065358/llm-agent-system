from fastapi import FastAPI
from pydantic import BaseModel

from hello_agents import HelloAgentsLLM, ToolRegistry
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
    return {"answer": result}
