from typing import List, Dict, Any, Optional


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",

    "reflect": """
请审查以下回答并找出问题:

# 原始任务
{task}

# 当前回答
{content}

请指出不足，并给出改进建议。
如果回答已经很好，只回答: 无需改进
""",

    "refine": """
请根据反馈改进回答:

# 原始任务
{task}

# 上一轮回答
{last_attempt}

# 反馈
{feedback}

请输出改进后的完整回答。
"""
}


class Memory:

    def __init__(self):
        self.records: List[Dict[str, str]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({
            "type": record_type,
            "content": content
        })

    def get_last_execution(self) -> Optional[str]:
        for r in reversed(self.records):
            if r["type"] == "execution":
                return r["content"]
        return None

    def get_trajectory(self) -> str:
        parts = []
        for r in self.records:

            if r["type"] == "execution":
                parts.append(f"[尝试]\n{r['content']}")

            if r["type"] == "reflection":
                parts.append(f"[反馈]\n{r['content']}")

        return "\n\n".join(parts)


class MyReflectionAgent:

    def __init__(
        self,
        llm_client,
        name="ReflectionAgent",
        max_iterations=1,
        custom_prompts=None
    ):

        self.llm_client = llm_client
        self.memory = Memory()
        self.name = name
        self.max_iterations = max_iterations
        self.prompts = custom_prompts or DEFAULT_PROMPTS

    def run(self, task: str):

        print(f"\n===== 任务开始 =====")
        print(task)

        # 初始生成
        attempt = self._llm(
            self.prompts["initial"].format(task=task)
        )

        self.memory.add_record("execution", attempt)

        for i in range(self.max_iterations):

            print(f"\n--- Iteration {i+1} ---")

            last = self.memory.get_last_execution()

            # 反思
            feedback = self._llm(
                self.prompts["reflect"].format(
                    task=task,
                    content=last
                )
            )

            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback:
                print("✓ 模型认为已完成")
                break

            # 优化
            improved = self._llm(
                self.prompts["refine"].format(
                    task=task,
                    last_attempt=last,
                    feedback=feedback
                )
            )

            self.memory.add_record("execution", improved)

        final = self.memory.get_last_execution()

        print("\n===== 最终结果 =====")
        print(final)

        return final

    def _llm(self, prompt: str) -> str:

        messages = [{"role": "user", "content": prompt}]
        result = self.llm_client.think(messages=messages)

        if not result:
            return ""

        if isinstance(result, str):
            return result.strip()

        try:
            return "".join(chunk for chunk in result if chunk).strip()
        except Exception:
            return str(result)
    def reflect(self, query: str, answer: str) -> str:
        prompt = f"""
    请审查以下回答：
    
    # 原始任务
    {query}
    
    # 当前回答
    {answer}
    
    请指出问题并给出改进建议。
    如果没有问题，只回答：无需改进
    """
        return self._llm(prompt)