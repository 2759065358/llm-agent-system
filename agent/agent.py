from dotenv import load_dotenv
load_dotenv()


class CodeAgent:

    def __init__(self, llm, tool_registry):

        self.llm = llm
        self.tool_registry = tool_registry

        self.memory_tool = tool_registry.get_tool("memory")
        self.rag_tool = tool_registry.get_tool("rag")

        from context.context_builder import MyContextBuilder
        self.context_builder = MyContextBuilder(
            memory_tool=self.memory_tool
        )

        from agent.react_agent import MyReActAgent
        self.react_agent = MyReActAgent(
            name="react",
            llm=llm,
            tool_registry=tool_registry
        )


        self.reflection_agent = None

    def run(self, query: str):

        original_query = query

        context = self.context_builder.build(query)

        result = self.react_agent.run(context)


        final = result

        # 3️⃣ 写入记忆
        try:
            self.memory_tool.run({
                "action": "add",
                "content": f"Q: {original_query}\nA: {final}"
            })
        except Exception as e:
            print(f"[WARNING] memory write failed: {e}")

        return {
                "final": final,
                # "trace": self.react_agent.current_history
            }
