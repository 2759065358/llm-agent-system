from typing import Optional, List
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.core.message import Message
from memory.memory_tool import MemoryTool
from rag.rag_tool import RAGTool
from dotenv import load_dotenv
load_dotenv()


class MyContextBuilder:

    def __init__(self, memory_tool=None, rag_pipeline=None):

        self.memory_tool = memory_tool if isinstance(memory_tool, MemoryTool) else None

        try:
            # 默认创建 RAGTool；允许外部注入自定义 rag_tool/pipeline（用于测试或替换实现）
            self.rag_tool = rag_pipeline if rag_pipeline is not None else RAGTool()
        except Exception:
            self.rag_tool = None


        self.builder = ContextBuilder(
            memory_tool=self.memory_tool,
            rag_tool=self.rag_tool,
            config=ContextConfig(
                max_tokens=4000,   
                min_relevance=0.2
            )
        )

    def build(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        构建上下文（稳定版）
        """

        
        safe_history = []
        if history:
            for h in history:
                if isinstance(h, Message):
                    safe_history.append(h)
                else:

                    safe_history.append(
                        Message(role="user", content=str(h))
                    )

        try:
            context = self.builder.build(
                user_query=query,
                conversation_history=safe_history,
                system_instructions=system_prompt
            )
        except Exception as e:

            context = f"{query}\n\n[Context构建失败: {e}]"


        return context if isinstance(context, str) else str(context)
