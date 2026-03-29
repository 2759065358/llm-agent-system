from dotenv import load_dotenv

load_dotenv()


class CodeAgent:

    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry

        self.memory_tool = tool_registry.get_tool("memory")

        from context.context_builder import MyContextBuilder
        self.context_builder = MyContextBuilder(memory_tool=self.memory_tool)

        from agent.planner_executor_agent import PlannerExecutorAgent
        self.core_agent = PlannerExecutorAgent(llm=llm, tool_registry=tool_registry)

    def run(self, query: str):
        # 将上下文注入 query，避免构建后未使用
        context = self.context_builder.build(query)
        composed_query = f"{query}\n\n[上下文]\n{context}" if context else query

        result = self.core_agent.run(composed_query)
        final = result.get("final", "")

        try:
            self.memory_tool.run({
                "action": "add",
                "content": {
                    "query": query,
                    "answer": final,
                }
            })
        except Exception as e:
            print(f"[WARNING] memory write failed: {e}")

        return result
