from __future__ import annotations

import unittest
import sys
import types
from types import SimpleNamespace

# Keep this unit test independent of the optional online API package.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from utils import Agent


class RecordingSink:
    def __init__(self):
        self.calls = []

    def record_agent_call(self, **call):
        self.calls.append(call)


class FakeCompletions:
    def create(self, **config):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="测试回答"))]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class AgentTraceTests(unittest.TestCase):
    def test_non_stream_agent_call_is_recorded(self):
        sink = RecordingSink()
        agent = Agent(
            FakeClient(),
            role_message="系统角色",
            role="招募者",
            trace_recorder=sink,
        )
        result = agent.chat("测试问题")
        self.assertEqual(result, "测试回答")
        self.assertEqual(len(sink.calls), 1)
        self.assertEqual(sink.calls[0]["role"], "招募者")
        self.assertEqual(sink.calls[0]["status"], "success")
        self.assertIn("测试问题", sink.calls[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
