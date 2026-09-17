"""MindLoop turn 中断测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from mind.loop import MindLoop, TurnInterruptedError, _InterruptToken
from mind.provider import LLMResponse, StreamEvent


class _FakeStreamProvider:
    """慢速 stream provider：每个 event 之间 sleep 0.05s，便于中断。"""

    def __init__(self, chunks: list[str], max_iters: int = 100) -> None:
        self.chunks = chunks
        self.max_iters = max_iters
        self.call_count = 0

    async def chat_stream(self, messages, tools=None):
        self.call_count += 1
        for chunk in self.chunks:
            await asyncio.sleep(0.02)
            yield StreamEvent(content=chunk)
        # 末尾给一个完整 response
        yield StreamEvent(
            content="",
            response=LLMResponse(
                content="".join(self.chunks),
                tool_calls=[],
                usage={"total_tokens": 10},
            ),
        )


def _make_mind_loop(workspace: Path, provider) -> MindLoop:
    """构造一个最小可用的 MindLoop（替换内部 provider）。"""
    config = MagicMock()
    config.workspace = workspace
    loop = MindLoop.__new__(MindLoop)
    # 手工注入依赖
    loop._config = config
    loop._session_id = "test-session"
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

    async def _fake_build_messages(_ctx):
        return []

    async def _fake_turn_committed(*_args, **_kwargs):
        return None

    loop._build_messages = _fake_build_messages
    loop._execute_tool_calls = MagicMock()
    loop._emit_turn_committed = _fake_turn_committed
    return loop


class InterruptTokenTest(unittest.TestCase):
    """_InterruptToken 单元。"""

    def test_default_not_requested(self) -> None:
        token = _InterruptToken(turn_id="abc")
        self.assertFalse(token.requested)

    def test_requested_flag(self) -> None:
        token = _InterruptToken(turn_id="abc")
        token.requested = True
        self.assertTrue(token.requested)


class InterruptBasicTest(unittest.IsolatedAsyncioTestCase):
    """MindLoop.interrupt_current 基础行为。"""

    def test_no_active_turn_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            self.assertFalse(loop.interrupt_current())

    def test_active_turn_marked_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            token = _InterruptToken(turn_id="x")
            loop._active_token = token
            self.assertTrue(loop.interrupt_current())
            self.assertTrue(token.requested)

    def test_double_interrupt_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            token = _InterruptToken(turn_id="x")
            loop._active_token = token
            self.assertTrue(loop.interrupt_current())
            # 第二次调用应返回 False（已请求过）
            self.assertFalse(loop.interrupt_current())

    def test_is_busy_reflects_active_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            self.assertFalse(loop.is_busy)
            loop._active_token = _InterruptToken(turn_id="x")
            self.assertTrue(loop.is_busy)


class CheckInterruptedTest(unittest.IsolatedAsyncioTestCase):
    """_check_interrupted 行为。"""

    def test_unrequested_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            token = _InterruptToken(turn_id="x")
            # 不应抛错
            loop._check_interrupted(token)

    def test_requested_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop = _make_mind_loop(Path(temp_dir), MagicMock())
            token = _InterruptToken(turn_id="abc")
            token.requested = True
            with self.assertRaises(TurnInterruptedError) as ctx:
                loop._check_interrupted(token)
            self.assertEqual(ctx.exception.turn_id, "abc")


class RunStreamInterruptTest(unittest.IsolatedAsyncioTestCase):
    """run_stream 在收到中断请求时抛出 TurnInterruptedError。"""

    async def test_interrupt_during_stream_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 100 个 chunk × 0.02s ≈ 2s，远大于 0.05s 的中断延迟，
            # 避免定时器精度导致 stream 先于中断结束
            provider = _FakeStreamProvider([f"c{i} " for i in range(100)])
            loop = _make_mind_loop(Path(temp_dir), provider)

            # 后台 task：0.05s 后发起中断
            async def _interrupt_after_delay():
                await asyncio.sleep(0.05)
                loop.interrupt_current()

            interrupt_task = asyncio.create_task(_interrupt_after_delay())
            with self.assertRaises(TurnInterruptedError):
                async for _chunk in loop.run_stream("test input"):
                    pass

            await interrupt_task
            # 验证 active_token 已被清理（finally 块）
            self.assertIsNone(loop._active_token)

    async def test_completion_clears_active_token(self) -> None:
        """正常完成的 turn 也应清理 active_token。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = _FakeStreamProvider(["OK"])
            loop = _make_mind_loop(Path(temp_dir), provider)

            chunks = []
            async for chunk in loop.run_stream("test"):
                chunks.append(chunk)
            self.assertEqual("".join(chunks), "OK")
            self.assertIsNone(loop._active_token)
            self.assertFalse(loop.is_busy)


if __name__ == "__main__":
    unittest.main()