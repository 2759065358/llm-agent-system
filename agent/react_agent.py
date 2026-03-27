import re
import json
import os
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, ToolRegistry
from hello_agents.tools.response import ToolResponse


STRICT_REACT_PROMPT = """你是一个严格遵循流程的文档分析助手。

## 可用工具
{tools}

## 策略规则（优先遵守）
1. 默认优先调用 rag 获取证据，再回答
2. 如果问题是常识、寒暄或无需外部信息，可直接 Finish
3. 最多调用一次 rag，避免循环
4. 每一步都必须输出 Thought 和 Action
5. 无论是否调用工具，最终都必须以 Finish 结束

输出必须严格使用以下英文格式（禁止中文替代）：
Thought: ...
Action: ...

工具调用格式：
- Action: rag[{{"query": "...", "top_k": 3}}]
- Action: Finish[最终答案]

严格禁止：
- 使用“动作”、“结尾”等中文替代 Action / Finish
- 输出解释性段落代替 Action
- 重复调用 rag

最终答案格式（仅在 Finish 中出现）
1. 结论
2. 文档依据
3. 分析过程
4. 信息缺口（如果有）

## 当前任务
{question}

## 历史
{history}

开始：
"""


class MyReActAgent(ReActAgent):

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        super().__init__(name, llm, system_prompt, config)

        self.tool_registry = tool_registry
        self.max_steps = int(os.getenv("AGENT_MAX_STEPS", "4"))
        self.current_history: List[str] = []
        self.used_rag = False
        self.prompt_template = STRICT_REACT_PROMPT

    def run(self, question: str) -> str:

        self.current_history = []
        self.used_rag = False

        for step in range(self.max_steps):

            print(f"\n🚀 STEP {step}")

            prompt = self.prompt_template.format(
                tools=self._render_tools(),
                question=question,
                history="\n".join(self.current_history)
            )

            output = self._llm(prompt)

            thought, action = self._parse(output)
            print("Action:", action)

            if not action:
                return "❌ LLM输出格式错误（无Action）"

            if "Finish[" in action:
                action = action.split("Finish[", 1)[1]
                action = action.rsplit("]", 1)[0]
                return action.strip()
            # ✅ 调工具
            result = self._call_tool(action)
            self.current_history.append(
                f"{output}\nObservation: {result}"
            )

        return f"❌ 未在{self.max_steps}步内完成任务"

    def _llm(self, prompt: str) -> str:


        messages = [{"role": "user", "content": prompt}]
        result = self.llm.think(messages=messages)


        if isinstance(result, str):
            return result.strip()

        try:
            return "".join(chunk for chunk in result if chunk).strip()
        except Exception:
            return str(result)

    def _parse(self, text: str) -> Tuple[str, str]:

        thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""

        return thought, action

    def _call_tool(self, action: str) -> str:
        match = re.match(r"(\w+)\[(.*)\]", action)
        if not match:
            return "❌ Action格式错误"

        tool_name, tool_input_str = match.groups()

        # 🔥 限制 rag 只能调用一次
        if tool_name == "rag":
            if self.used_rag:
                return "⚠️ 已调用过RAG，禁止重复调用"
            self.used_rag = True

        try:
            tool_input = json.loads(tool_input_str) if tool_input_str else {}
        except Exception:
            tool_input = {"query": tool_input_str}

        response = self.tool_registry.execute_tool(
            tool_name,
            json.dumps(tool_input, ensure_ascii=False)
        )
        if isinstance(response, ToolResponse):
            observation = {
                "status": response.status.value,
                "text": response.text,
                "data": response.data or {}
            }
            return json.dumps(observation, ensure_ascii=False)
        return str(response)

    def _render_tools(self) -> str:

        tools = []
        for name, tool in getattr(self.tool_registry, "_tools", {}).items():
            tools.append(f"{name}: {tool.description}")

        return "\n".join(tools)
