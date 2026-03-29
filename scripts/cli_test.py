from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.registry import ToolRegistry
from memory.memory_tool import MemoryTool
from rag.rag_tool import RAGTool
from agent.agent import CodeAgent


# ✅ 1. 初始化 LLM
llm = HelloAgentsLLM()

# ✅ 2. 初始化工具注册器
tool_registry = ToolRegistry()

# ✅ 3. 注册工具
tool_registry.register_tool(MemoryTool())
tool_registry.register_tool(RAGTool())

# ✅ 4. 创建 Agent（关键）
agent = CodeAgent(llm, tool_registry)

while True:
    query = input("请输入问题")


    result = agent.run(query)
    print(result)
