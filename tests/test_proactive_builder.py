"""主动链路总装配测试：builder 组装 kernel + loop 端到端。"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from initiative.data_sources import DataSourceManager
from initiative.proactive_builder import (
    ProactiveEngineDeps,
    _session_key,
    build_proactive_engine,
)
from initiative.proactive_emotion_module import SLOT_PROMPT_SECTION as EMOTION_SECTION
from initiative.proactive_feedback_module import SLOT_FEEDBACK_STATS
from initiative.proactive_gateway import DataGateway, GatewayResult
from initiative.proactive_schedule import SLOT_NEXT_INTERVAL


def _cfg(**overrides):
    base = {
        "default_channel": "telegram",
        "default_chat_id": "12345",
        "agent_tick_content_limit": 3,
        # scheduler 参数
        "interval_seconds": 1800,
        "tick_interval_s0": 4800,
        "tick_interval_s1": 2400,
        "tick_jitter": 0.0,
        "score_weight_energy": 0.40,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _TmpDir:
    def __init__(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name)

    def cleanup(self):
        self._tmp.cleanup()


class SessionKeyTest(unittest.TestCase):
    def test_combines_channel_chat(self) -> None:
        self.assertEqual(_session_key(_cfg()), "telegram:12345")

    def test_missing_chat(self) -> None:
        self.assertEqual(_session_key(_cfg(default_chat_id="")), "")

    def test_missing_channel(self) -> None:
        self.assertEqual(_session_key(_cfg(default_channel=" ")), "")


class MinimalBuildTest(unittest.IsolatedAsyncioTestCase):
    """零依赖装配：只有昼夜 gate + schedule。"""

    def test_returns_kernel_and_loop(self) -> None:
        kernel, loop = build_proactive_engine(_cfg())
        from initiative.proactive_kernel import ProactiveKernel
        from initiative.proactive_loop import ProactiveLoop

        self.assertIsInstance(kernel, ProactiveKernel)
        self.assertIsInstance(loop, ProactiveLoop)

    def test_module_count_minimal(self) -> None:
        kernel, _ = build_proactive_engine(_cfg())
        # 昼夜 gate + schedule = 2
        self.assertEqual(len(kernel._all_modules()), 2)

    async def test_minimal_tick_runs(self) -> None:
        kernel, loop = build_proactive_engine(_cfg())
        score = await kernel.run_tick("telegram:12345")
        # 无 judge 输出 → None，但 schedule 写了间隔槽
        self.assertIsNone(score)

    def test_loop_session_key(self) -> None:
        _, loop = build_proactive_engine(_cfg())
        self.assertEqual(loop._session_key, "telegram:12345")


class FullBuildTest(unittest.IsolatedAsyncioTestCase):
    """全依赖装配：昼夜 + source + 情绪 + 反馈 + schedule。"""

    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.ds = DataSourceManager()
        self.ds.content_source._score_threshold = 0.0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _deps(self, **extra):
        return ProactiveEngineDeps(
            data_sources=self.ds,
            emotion_db_path=self.tmp.path / "emotion.db",
            feedback_db_path=self.tmp.path / "fb.db",
            **extra,
        )

    def test_all_modules_present(self) -> None:
        kernel, _ = build_proactive_engine(_cfg(), self._deps())
        slots = {phase: mods for phase, mods in kernel._runner.all_slots().items()
                 if mods}
        phases_with_modules = set(slots)
        self.assertIn("proactive.gate", phases_with_modules)
        self.assertIn("proactive.source", phases_with_modules)
        self.assertIn("proactive.prompt", phases_with_modules)
        self.assertIn("proactive.schedule", phases_with_modules)

    async def test_full_tick_writes_slots(self) -> None:
        self.ds.alert_source.add_alert("测试告警")
        kernel, _ = build_proactive_engine(_cfg(), self._deps())
        from initiative.proactive_frame import new_frame

        # 只跑一轮（alert 队列 fetch 即清空，不可重复消费）
        frame = await kernel._runner.run(new_frame("telegram:12345"))
        self.assertEqual(len(frame.slots["proactive:alerts"]), 1)
        self.assertIn(EMOTION_SECTION, frame.slots)  # 情绪区块
        self.assertIn(SLOT_FEEDBACK_STATS, frame.slots)  # 反馈统计
        self.assertIn(SLOT_NEXT_INTERVAL, frame.slots)  # 调度间隔

    async def test_extra_module_appended(self) -> None:
        class FakeJudge:
            slot = "judge"
            phase = "proactive.judge"

            async def run(self, frame):
                from initiative.proactive_frame import ProactiveTickResult

                frame.output = ProactiveTickResult(base_score=0.66)
                return frame

        deps = ProactiveEngineDeps(extra_modules=[FakeJudge()])
        kernel, _ = build_proactive_engine(_cfg(), deps)
        self.assertEqual(len(kernel._all_modules()), 3)
        score = await kernel.run_tick("s")
        self.assertAlmostEqual(score, 0.66)

    async def test_gateway_snapshot_flows(self) -> None:
        """不带 data_sources 时 gateway 快照进入 initial slots。"""
        gw = DataGateway()

        async def fake_run():
            return GatewayResult(alerts=[{"id": "g1"}])

        gw.run = fake_run  # type: ignore[assignment]
        seen: dict = {}

        class Capture:
            slot = "capture"
            phase = "proactive.gate"

            async def run(self, frame):
                seen.update(frame.slots)
                return frame

        deps = ProactiveEngineDeps(gateway=gw, extra_modules=[Capture()])
        kernel, _ = build_proactive_engine(_cfg(), deps)
        await kernel.run_tick("s")  # run_tick 内做 gateway 并入
        self.assertEqual(seen["proactive:alerts"], [{"id": "g1"}])

    async def test_loop_with_engine_runs(self) -> None:
        """假时钟驱动装配出的 loop 跑两轮。"""

        class _Clock:
            def __init__(self):
                self.sleeps: list[float] = []

            async def sleep(self, s):
                self.sleeps.append(s)
                await asyncio.sleep(0)

        clock = _Clock()
        _, loop = build_proactive_engine(_cfg(), self._deps())
        loop._sleep = clock.sleep
        loop._max_ticks = 2
        await loop.run()
        self.assertEqual(loop.tick_count, 2)
        self.assertEqual(len(clock.sleeps), 2)


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_kernel_start_stop_with_builder(self) -> None:
        started: list[str] = []

        class WithLifecycle:
            slot = "lifecycle_probe"
            phase = "proactive.gate"

            async def start(self):
                started.append("start")

            async def stop(self):
                started.append("stop")

            async def run(self, frame):
                return frame

        deps = ProactiveEngineDeps(extra_modules=[WithLifecycle()])
        kernel, _ = build_proactive_engine(_cfg(), deps)
        await kernel.start()
        await kernel.stop()
        self.assertEqual(started, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
