import json
import os
import unittest

from agent.react_agent import MyReActAgent
from hello_agents.tools.response import ToolResponse


class _DummyLLM:
    model = "dummy"

    def think(self, messages):
        return "Thought: done\nAction: Finish[ok]"


class _DummyRegistry:
    _tools = {}

    def execute_tool(self, name, input_text):
        return ToolResponse.success(text="tool text", data={"key": "value"})


class TestReActAgent(unittest.TestCase):
    def test_max_steps_from_env(self):
        old = os.environ.get("AGENT_MAX_STEPS")
        os.environ["AGENT_MAX_STEPS"] = "6"
        try:
            agent = MyReActAgent("react", _DummyLLM(), _DummyRegistry())
            self.assertEqual(agent.max_steps, 6)
        finally:
            if old is None:
                os.environ.pop("AGENT_MAX_STEPS", None)
            else:
                os.environ["AGENT_MAX_STEPS"] = old

    def test_call_tool_keeps_structured_data(self):
        agent = MyReActAgent("react", _DummyLLM(), _DummyRegistry())
        observation = agent._call_tool('memory[{"action":"search","query":"x"}]')
        payload = json.loads(observation)

        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["data"]["key"], "value")


if __name__ == "__main__":
    unittest.main()
