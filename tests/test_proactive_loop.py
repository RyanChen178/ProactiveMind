"""主动链路主循环测试：假时钟驱动的 sleep→tick 序列。"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from initiative.proactive_frame import new_frame
from initiative.proactive_loop import ProactiveLoop
from initiative.proactive_phases import ProactivePhaseRunner
from initiative.proactive_schedule import (
    SLOT_BASE_SCORE,
    ProactiveScheduler,
)


def _scheduler(interval=100):
    """固定间隔调度器（简单 stub）。"""
    return SimpleNamespace(next_interval=lambda score=None: interval)


class _FakeClock:
    """收集 sleep 调用并立即返回（不真睡）。"""

    def __init__(self):
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


def _runner_with_score(score=None):
    """每次 tick 写入固定 base_score 的 runner stub。"""
    class ScoreModule:
        slot = "judge"
        phase = "proactive.judge"

        def __init__(self, score):
            self._score = score

        async def run(self, frame):
            if self._score is not None:
                frame.slots[SLOT_BASE_SCORE] = self._score
            return frame

    return ProactivePhaseRunner([ScoreModule(score)])


class RunSequenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_sleep_then_tick_order(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(0.5),
            scheduler=_scheduler(120),
            sleep_fn=clock.sleep,
            max_ticks=3,
        )
        await loop.run()
        # 3 轮 tick → 3 次 sleep（每轮先睡再 tick）
        self.assertEqual(len(clock.sleeps), 3)
        self.assertTrue(all(s == 120 for s in clock.sleeps))
        self.assertEqual(loop.tick_count, 3)

    async def test_first_interval_override(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(None),
            scheduler=_scheduler(999),
            sleep_fn=clock.sleep,
            max_ticks=2,
            first_interval_s=5,
        )
        await loop.run()
        # 第一轮用 first_interval，第二轮走 scheduler（score=None）
        self.assertEqual(clock.sleeps[0], 5)
        self.assertEqual(clock.sleeps[1], 999)

    async def test_score_drives_next_interval(self) -> None:
        """分数传给调度器：高分 → 短间隔。"""
        intervals_by_score: list[float | None] = []

        def _next(score=None):
            intervals_by_score.append(score)
            return 10

        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(0.8),
            scheduler=SimpleNamespace(next_interval=_next),
            sleep_fn=clock.sleep,
            max_ticks=3,
        )
        await loop.run()
        # 第一轮无分数（None），后两轮拿到 0.8
        self.assertIsNone(intervals_by_score[0])
        self.assertEqual(intervals_by_score[1], 0.8)
        self.assertEqual(intervals_by_score[2], 0.8)

    async def test_tick_exception_continues(self) -> None:
        clock = _FakeClock()
        errors: list[BaseException] = []

        class BoomRunner:
            async def run(self, frame):
                raise RuntimeError("tick 挂了")

        loop = ProactiveLoop(
            runner=BoomRunner(),  # type: ignore[arg-type]
            scheduler=_scheduler(10),
            sleep_fn=clock.sleep,
            on_tick_error=errors.append,
            max_ticks=3,
        )
        await loop.run()
        # 异常被捕获、循环没终止
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(clock.sleeps), 3)
        self.assertFalse(loop.is_running)  # max_ticks 正常收尾

    async def test_stop_breaks_after_sleep(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(10),
            sleep_fn=clock.sleep,
        )

        async def _stop_after_two_sleeps(seconds: float) -> None:
            clock.sleeps.append(seconds)
            await asyncio.sleep(0)
            if len(clock.sleeps) >= 2:
                loop.stop()

        loop._sleep = _stop_after_two_sleeps
        await loop.run()
        # 第二次睡醒后 stop → 不再有第三次 tick
        self.assertEqual(loop.tick_count, 1)

    async def test_max_ticks_one(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(1),
            sleep_fn=clock.sleep,
            max_ticks=1,
        )
        await loop.run()
        self.assertEqual(loop.tick_count, 1)
        self.assertFalse(loop.is_running)


class TickFnOverrideTest(unittest.IsolatedAsyncioTestCase):
    async def test_tick_fn_replaces_runner(self) -> None:
        clock = _FakeClock()
        seen_frames: list[ProactiveFrame] = []

        async def _custom_tick(frame):
            seen_frames.append(frame)
            frame.slots[SLOT_BASE_SCORE] = 0.9
            return frame

        loop = ProactiveLoop(
            runner=_runner_with_score(0.1),  # 应被 tick_fn 覆盖
            scheduler=_scheduler(10),
            sleep_fn=clock.sleep,
            tick_fn=_custom_tick,
            max_ticks=1,
        )
        await loop.run()
        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(loop.tick_count, 1)

    async def test_tick_fn_exception_captured(self) -> None:
        clock = _FakeClock()
        errors: list[BaseException] = []

        async def _bad_tick(frame):
            raise ValueError("自定义 tick 失败")

        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(5),
            sleep_fn=clock.sleep,
            tick_fn=_bad_tick,
            on_tick_error=errors.append,
            max_ticks=1,
        )
        await loop.run()
        self.assertEqual(len(errors), 1)


class StateTest(unittest.IsolatedAsyncioTestCase):
    async def test_not_running_before_run(self) -> None:
        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(1),
        )
        self.assertFalse(loop.is_running)

    async def test_running_flipped_off_after_finish(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(1),
            sleep_fn=clock.sleep,
            max_ticks=1,
        )
        await loop.run()
        self.assertFalse(loop.is_running)

    async def test_tick_count_resets_each_run(self) -> None:
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=_scheduler(1),
            sleep_fn=clock.sleep,
            max_ticks=2,
        )
        await loop.run()
        self.assertEqual(loop.tick_count, 2)
        await loop.run()
        self.assertEqual(loop.tick_count, 2)

    async def test_next_interval_delegates_scheduler(self) -> None:
        calls: list = []

        def _next(score=None):
            calls.append(score)
            return 42

        loop = ProactiveLoop(
            runner=_runner_with_score(),
            scheduler=SimpleNamespace(next_interval=_next),
        )
        self.assertEqual(loop.next_interval(0.7), 42)
        self.assertEqual(calls, [0.7])


class IntegrationTest(unittest.IsolatedAsyncioTestCase):
    """真模块全链路：schedule 壳 + loop 装配。"""

    async def test_real_schedule_module_in_loop(self) -> None:
        """schedule 模块挂进 runner，loop 每轮把间隔写进 frame。"""
        from types import SimpleNamespace
        from initiative.proactive_schedule import (
            SLOT_NEXT_INTERVAL,
            ProactiveScheduleModule,
        )

        sched = ProactiveScheduler(
            cfg=SimpleNamespace(
                interval_seconds=1800,
                tick_interval_s0=4800,
                tick_interval_s1=2400,
                tick_jitter=0.0,
                score_weight_energy=0.40,
            ),
            presence=None,  # 走固定间隔档
        )
        runner = ProactivePhaseRunner([ProactiveScheduleModule(sched)])
        clock = _FakeClock()
        loop = ProactiveLoop(
            runner=runner,
            scheduler=_scheduler(7),
            sleep_fn=clock.sleep,
            max_ticks=2,
        )
        await loop.run()
        # 两轮 tick 的 sleep 都是 stub 的 7（调度独立）
        self.assertEqual(clock.sleeps, [7, 7])


if __name__ == "__main__":
    unittest.main()
