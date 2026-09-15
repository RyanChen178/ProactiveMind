"""MindLoop 接入 trace_span 集成测试。

验证每次 turn 自动产生 trace_id 并贯穿日志。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

from core.diagnostics import (
    JsonFormatter,
    LogCapture,
    configure_json_logging,
    current_trace_id,
    new_trace_id,
    reset_for_test,
    trace_span,
)
from mind.loop import MindLoop
from mind.provider import LLMResponse, StreamEvent


class _FakeStreamProvider:
    """简单 provider：chat 和 chat_stream 都返回响应。"""

    def __init__(self, content: str = "ok", tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    async def chat(self, messages, tools=None):
        return LLMResponse(
            content=self.content,
            tool_calls=self.tool_calls,
            usage={"total_tokens": 10},
        )

    async def chat_stream(self, messages, tools=None):
        for _ in range(1):
            yield StreamEvent(
                content=self.content,
                response=LLMResponse(
                    content=self.content,
                    tool_calls=self.tool_calls,
                    usage={"total_tokens": 10},
                ),
            )


def _make_mind_loop(provider) -> MindLoop:
    """构造最小 MindLoop 实例（不依赖完整 Config）。"""
    config = MagicMock()
    config.workspace = Path(tempfile.gettempdir()) / "proactivemind-trace-test"
    config.workspace.mkdir(parents=True, exist_ok=True)

    loop = MindLoop.__new__(MindLoop)
    loop._config = config
    loop._session_id = "trace-test-session"
    loop._session = MagicMock()
    loop._session.add_user = MagicMock()
    loop._session.add_assistant = MagicMock()
    loop._presence = None
    loop._bus = MagicMock()
    loop._extension_manager = None
    loop._tools = MagicMock()
    loop._tools.get_schemas = MagicMock(return_value=[])
    loop._stats = MagicMock()
    loop._stats.record = MagicMock()
    loop._active_token = None
    loop._active_task = None
    loop._provider = provider
    loop._prompts = MagicMock()
    loop._prompts.build = MagicMock(return_value="system prompt")

    async def _fake_build_messages(_ctx):
        return []

    async def _fake_emit_turn_committed(*_args, **_kwargs):
        return None

    loop._build_messages = _fake_build_messages
    loop._execute_tool_calls = MagicMock()
    loop._emit_turn_committed = _fake_emit_turn_committed
    return loop


class TurnTraceIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """MindLoop.run 跑一次后日志包含 trace_id。"""

    async def test_log_records_carry_trace_id(self) -> None:
        reset_for_test()
        provider = _FakeStreamProvider("hello")
        mloop = _make_mind_loop(provider)

        buf = StringIO()
        import logging

        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("event", "action", "step_count")))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)  # 临时降低门槛
        try:
            result = await mloop.run("hi")
        finally:
            root.removeHandler(handler)

        self.assertEqual(result, "hello")

        # 解析所有日志行
        events: list[dict] = []
        for line in buf.getvalue().strip().splitlines():
            if line.strip():
                events.append(json.loads(line))

        # 至少有 trace_id 字段
        events_with_trace = [e for e in events if "trace_id" in e]
        self.assertGreater(len(events_with_trace), 0, f"events={events}")

        # 同一 turn 的日志共享同一个 trace_id
        trace_ids = {e["trace_id"] for e in events_with_trace}
        self.assertEqual(len(trace_ids), 1)

        # 应包含 turn_started 和 turn_finished 事件
        event_names = {e.get("event") for e in events_with_trace}
        self.assertIn("turn_started", event_names)
        self.assertIn("turn_finished", event_names)

    async def test_three_turns_produce_three_distinct_traces(self) -> None:
        """连续 3 次 turn 应产生 3 个不同的 trace_id。"""
        reset_for_test()
        buf = StringIO()
        import logging

        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("event",)))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            for i in range(3):
                provider = _FakeStreamProvider(f"r{i}")
                mloop = _make_mind_loop(provider)
                await mloop.run(f"turn {i}")
        finally:
            root.removeHandler(handler)

        events = []
        for line in buf.getvalue().strip().splitlines():
            if line.strip():
                events.append(json.loads(line))

        turn_started = [e for e in events if e.get("event") == "turn_started"]
        self.assertEqual(len(turn_started), 3)
        trace_ids = {e["trace_id"] for e in turn_started}
        self.assertEqual(len(trace_ids), 3)


class NestedSpanTest(unittest.IsolatedAsyncioTestCase):
    """嵌套 trace_span 共享 trace_id。"""

    def test_nested_shares_trace(self) -> None:
        reset_for_test()
        with trace_span("outer") as outer_info:
            outer_trace = current_trace_id()
            with trace_span("inner"):
                inner_trace = current_trace_id()
            # inner 退出后 outer 仍生效
            self.assertEqual(current_trace_id(), outer_trace)
        self.assertEqual(outer_trace, inner_trace)


class BindContextPropagationTest(unittest.IsolatedAsyncioTestCase):
    """bind_context 字段自动出现在日志中。"""

    def test_bind_context_in_json_output(self) -> None:
        reset_for_test()
        from core.diagnostics import bind_context, get_logger

        buf = StringIO()
        import logging

        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("session", "flow")))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            with bind_context(flow="probe", session="s-1"):
                get_logger("test.bind").info(
                    "inside",
                    extra={"event": "probe_started"},
                )
        finally:
            root.removeHandler(handler)

        lines = [line for line in buf.getvalue().strip().splitlines() if line.strip()]
        self.assertGreater(len(lines), 0)
        data = json.loads(lines[-1])
        self.assertEqual(data["flow"], "probe")
        self.assertEqual(data["session"], "s-1")
        self.assertEqual(data["event"], "probe_started")
        self.assertIn("trace_id", data)


if __name__ == "__main__":
    unittest.main()