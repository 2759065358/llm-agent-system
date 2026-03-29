from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json


PLANNER_PROMPT = """你是任务规划器。请根据用户问题输出执行计划。

要求：
1) 优先使用rag工具做检索，再综合回答。
2) 返回严格 JSON 数组，每个元素包含:
   - step_id: int
   - type: tool 或 llm
   - instruction: string
   - tool_name: string(当type=tool时)
   - tool_input: object(当type=tool时)

用户问题：{query}
可用工具：{tools}
"""


@dataclass
class PlanStep:
    step_id: int
    type: str
    instruction: str
    tool_name: str = ""
    tool_input: Dict[str, Any] | None = None


class Planner:
    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry

    def _tool_desc(self) -> str:
        lines = []
        for name, tool in getattr(self.tool_registry, "_tools", {}).items():
            lines.append(f"- {name}: {getattr(tool, 'description', '')}")
        return "\n".join(lines)

    def make_plan(self, query: str) -> List[PlanStep]:
        prompt = PLANNER_PROMPT.format(query=query, tools=self._tool_desc())
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm.think(messages=messages)
        text = raw if isinstance(raw, str) else "".join(chunk for chunk in raw if chunk)

        # 容错：若无法解析，使用默认两步计划
        try:
            data = json.loads(text)
            steps = []
            for item in data:
                steps.append(
                    PlanStep(
                        step_id=int(item.get("step_id", len(steps) + 1)),
                        type=item.get("type", "llm"),
                        instruction=item.get("instruction", ""),
                        tool_name=item.get("tool_name", ""),
                        tool_input=item.get("tool_input", {}) or {},
                    )
                )
            if steps:
                return steps
        except Exception:
            pass

        return [
            PlanStep(
                step_id=1,
                type="tool",
                instruction="先检索相关知识",
                tool_name="rag",
                tool_input={"action": "search", "query": query, "top_k": 5, "enable_rerank": True},
            ),
            PlanStep(
                step_id=2,
                type="llm",
                instruction="基于检索结果给出最终回答，包含结论和依据",
            )
        ]


class Executor:
    def __init__(self, llm, tool_registry):
        self.llm = llm
        self.tool_registry = tool_registry

    def run(self, query: str, plan: List[PlanStep]) -> Dict[str, Any]:
        observations: List[Dict[str, Any]] = []
        final_answer = ""

        for step in plan:
            if step.type == "tool":
                tool_input = step.tool_input or {}
                if step.tool_name == "rag" and "query" not in tool_input:
                    tool_input["query"] = query
                result = self.tool_registry.execute_tool(
                    step.tool_name,
                    json.dumps(tool_input, ensure_ascii=False)
                )
                observations.append({
                    "step_id": step.step_id,
                    "type": "tool",
                    "tool_name": step.tool_name,
                    "tool_input": tool_input,
                    "observation": result,
                })
                continue

            context = "\n".join(
                [f"Step {o['step_id']}({o.get('tool_name', o['type'])}): {o['observation']}" for o in observations]
            )
            prompt = (
                f"用户问题：{query}\n\n"
                f"执行上下文：\n{context}\n\n"
                f"当前任务：{step.instruction}\n"
                "请输出最终回答，尽量简洁并给出依据。"
            )
            messages = [{"role": "user", "content": prompt}]
            out = self.llm.think(messages=messages)
            final_answer = out if isinstance(out, str) else "".join(chunk for chunk in out if chunk)
            observations.append({
                "step_id": step.step_id,
                "type": "llm",
                "instruction": step.instruction,
                "observation": final_answer,
            })

        if not final_answer and observations:
            final_answer = str(observations[-1].get("observation", ""))

        return {
            "final": final_answer.strip(),
            "trace": observations,
        }


class PlannerExecutorAgent:
    def __init__(self, llm, tool_registry):
        self.planner = Planner(llm, tool_registry)
        self.executor = Executor(llm, tool_registry)

    def run(self, query: str) -> Dict[str, Any]:
        plan = self.planner.make_plan(query)
        result = self.executor.run(query, plan)
        result["plan"] = [step.__dict__ for step in plan]
        return result
