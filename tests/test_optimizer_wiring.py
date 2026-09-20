"""Optimizer 接线测试：config / MindLoop 生命周期 / CLI 命令。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mind.config import ConsolidationConfig, load_config
from mind.optimizer import MemoryOptimizer, MemoryOptimizerLoop


def _run(coro):
    return asyncio.run(coro)


def _write_config(temp_dir: str, extra: str = "") -> str:
    path = Path(temp_dir) / "config.toml"
    path.write_text(
        """
[llm]
main = "rt"

[llm.runtimes.rt]
provider = "openai"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"

[workspace]
path = "~/.proactivemind/workspace-optimizer-test"
"""
        + extra,
        encoding="utf-8",
    )
    return str(path)


class OptimizerIntervalConfigTest(unittest.TestCase):
    """[consolidation].optimizer_interval_seconds 解析。"""

    def test_default_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = load_config(_write_config(temp_dir))
            self.assertEqual(cfg.consolidation.optimizer_interval_seconds, 64800)

    def test_custom_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = load_config(_write_config(temp_dir, """
[consolidation]
enabled = true
optimizer_interval_seconds = 3600
"""))
            self.assertEqual(cfg.consolidation.optimizer_interval_seconds, 3600)


class OptimizerLoopStopTest(unittest.IsolatedAsyncioTestCase):
    """MemoryOptimizerLoop.stop() 行为。"""

    async def test_stop_exits_loop(self) -> None:
        optimizer = MagicMock()
        optimizer.optimize = AsyncMock(return_value=0)
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=300)
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.01)
        self.assertTrue(loop.is_running)
        loop.stop()
        await asyncio.wait_for(task, timeout=2.0)
        self.assertFalse(loop.is_running)
        # 停止后 optimizer 不应再被调用
        optimizer.optimize.assert_not_called()

    async def test_loop_runs_optimizer_on_tick(self) -> None:
        """把 interval 对齐到极短边界，验证 optimize 被调用。"""
        optimizer = MagicMock()
        calls = {"n": 0}

        async def _fake_optimize():
            calls["n"] += 1
            # 首次触发后把下一次 tick 推远，避免连续触发导致多次调用
            if calls["n"] == 1:
                loop._seconds_until_next_tick = lambda: 60.0  # type: ignore[assignment]
            return 3

        optimizer.optimize = _fake_optimize
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=300)
        loop._seconds_until_next_tick = lambda: 0.0  # type: ignore[assignment]
        task = asyncio.create_task(loop.run())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if calls["n"] >= 1:
                break
        loop.stop()
        await asyncio.wait_for(task, timeout=2.0)
        self.assertEqual(calls["n"], 1)


class MindLoopOptimizerWiringTest(unittest.IsolatedAsyncioTestCase):
    """MindLoop 持有 optimizer 并管理后台循环。"""

    def _make_agent(self, workspace: Path):
        from mind.loop import MindLoop

        agent = MindLoop.__new__(MindLoop)
        agent._config = MagicMock()
        agent._config.workspace = workspace
        agent._config.consolidation = ConsolidationConfig(
            optimizer_interval_seconds=300,
        )
        return agent

    async def test_init_builds_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            agent = self._make_agent(workspace)
            # 复刻 __init__ 中的构造段（避免完整依赖注入）
            from mind.optimizer import MemoryOptimizer, MemoryOptimizerLoop

            agent._optimizer = MemoryOptimizer(workspace)
            agent._optimizer_loop = MemoryOptimizerLoop(
                agent._optimizer, interval_seconds=300
            )
            agent._optimizer_task = None

            self.assertIsInstance(agent._optimizer, MemoryOptimizer)
            self.assertEqual(agent._optimizer._pending_file, workspace / "PENDING.md")

    async def test_start_optimizer_loop_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = self._make_agent(Path(temp_dir))
            agent._optimizer = MagicMock()
            agent._optimizer_loop = MagicMock()
            agent._optimizer_loop.run = _run_forever_fn  # async 函数，可重复调用
            agent._optimizer_task = None

            t1 = agent.start_optimizer_loop()
            t2 = agent.start_optimizer_loop()
            self.assertIs(t1, t2)
            self.assertFalse(t1.done())
            t1.cancel()
            try:
                await t1
            except asyncio.CancelledError:
                pass

    async def test_run_optimizer_now_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = self._make_agent(Path(temp_dir))
            agent._optimizer = MagicMock()
            agent._optimizer.optimize = AsyncMock(return_value=5)
            agent._optimizer_loop = MagicMock()
            agent._optimizer_task = None

            result = await agent.run_optimizer_now()
            self.assertEqual(result, 5)
            agent._optimizer.optimize.assert_called_once()


async def _run_forever_fn() -> None:
    """可重复调用的无限 async 函数（供 mock 循环）。"""
    while True:
        await asyncio.sleep(1)


class OptimizeCommandTest(unittest.TestCase):
    """CLI /optimize 命令。"""

    def test_reports_archived_count(self) -> None:
        from cli_commands import CommandContext, dispatch

        agent = MagicMock()
        agent.run_optimizer_now = AsyncMock(return_value=4)
        output: list[str] = []
        ctx = CommandContext(agent=agent, output=lambda s: output.append(s))
        _run(dispatch(ctx, "/optimize"))
        self.assertIn("已归档 4 条", output[0])

    def test_reports_zero(self) -> None:
        from cli_commands import CommandContext, dispatch

        agent = MagicMock()
        agent.run_optimizer_now = AsyncMock(return_value=0)
        output: list[str] = []
        ctx = CommandContext(agent=agent, output=lambda s: output.append(s))
        _run(dispatch(ctx, "/optimize"))
        self.assertIn("没有可归档", output[0])

    def test_unavailable_optimizer(self) -> None:
        from cli_commands import CommandContext, dispatch

        agent = MagicMock(spec=[])  # 无 run_optimizer_now 属性
        output: list[str] = []
        ctx = CommandContext(agent=agent, output=lambda s: output.append(s))
        _run(dispatch(ctx, "/optimize"))
        self.assertIn("不可用", output[0])

    def test_optimize_registered(self) -> None:
        from cli_commands import known_commands

        self.assertIn("optimize", known_commands())


class OptimizerEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """run_optimizer_now 端到端：PENDING 归档进 MEMORY。"""

    async def test_archives_pending_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            optimizer = MemoryOptimizer(workspace)
            (workspace / "PENDING.md").write_text(
                "# 待归档\n\n- 用户长期偏好简洁的回答风格\n- 临时的短内容\n",
                encoding="utf-8",
            )
            archived = await optimizer.optimize()
            self.assertGreaterEqual(archived, 1)
            memory = (workspace / "MEMORY.md").read_text(encoding="utf-8")
            self.assertIn("简洁的回答风格", memory)


if __name__ == "__main__":
    unittest.main()