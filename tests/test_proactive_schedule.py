"""主动链路调度模块测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from initiative.proactive_frame import new_frame
from initiative.proactive_schedule import (
    SLOT_BASE_SCORE,
    SLOT_NEXT_INTERVAL,
    ProactiveScheduleModule,
    ProactiveScheduler,
)


def _make_cfg(**overrides):
    base = {
        "interval_seconds": 1800,
        "tick_interval_s0": 4800,
        "tick_interval_s1": 2400,
        "tick_jitter": 0.0,
        "score_weight_energy": 0.40,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_presence(last_user_at=None):
    return SimpleNamespace(get_last_user_at=lambda key: last_user_at)


def _collect_traces():
    traces: list[dict] = []
    return traces, (lambda **kw: traces.append(kw))


class NoPresenceTest(unittest.TestCase):
    def test_fixed_interval(self) -> None:
        traces, trace = _collect_traces()
        sched = ProactiveScheduler(
            cfg=_make_cfg(interval_seconds=900), presence=None, trace_fn=trace,
        )
        interval = sched.next_interval()
        self.assertEqual(interval, 900)
        self.assertEqual(traces[-1]["mode"], "fixed_no_presence")

    def test_base_score_ignored_without_presence(self) -> None:
        sched = ProactiveScheduler(
            cfg=_make_cfg(interval_seconds=600), presence=None,
        )
        # 无 presence 时即便给了分数也走固定间隔
        self.assertEqual(sched.next_interval(base_score=0.9), 600)


class DeriveScoreTest(unittest.TestCase):
    def test_score_derived_from_presence(self) -> None:
        # 用户 6 小时未活跃 → energy 低 → d_energy 高 → 分数较高 → 短间隔
        traces, trace = _collect_traces()
        sched = ProactiveScheduler(
            cfg=_make_cfg(),
            presence=_make_presence(
                datetime.now(timezone.utc) - timedelta(hours=6)
            ),
            target_session_key_fn=lambda: "web:u1",
            trace_fn=trace,
        )
        interval = sched.next_interval()
        # 分数 = (1-energy) * 0.4；6h 无活跃 energy≈0.01 → 分数≈0.4 > 0.20
        # → 短间隔档（jitter=0 → 精确 2400）
        self.assertEqual(interval, 2400)
        self.assertEqual(traces[-1]["mode"], "adaptive")
        self.assertGreater(traces[-1]["base_score"], 0.20)

    def test_fresh_user_uses_long_interval(self) -> None:
        # 用户刚活跃 → energy 高 → 分数低 → 长间隔
        sched = ProactiveScheduler(
            cfg=_make_cfg(),
            presence=_make_presence(datetime.now(timezone.utc)),
        )
        interval = sched.next_interval()
        self.assertEqual(interval, 4800)

    def test_explicit_score_wins(self) -> None:
        traces, trace = _collect_traces()
        sched = ProactiveScheduler(
            cfg=_make_cfg(),
            presence=_make_presence(),
            trace_fn=trace,
        )
        # 显式高分 → 不查 presence，直接短间隔
        self.assertEqual(sched.next_interval(base_score=0.5), 2400)
        # 低分 → 长间隔
        self.assertEqual(sched.next_interval(base_score=0.1), 4800)

    def test_weight_scales_score(self) -> None:
        # 权重 = 0.0 → 分数恒 0 → 永远长间隔
        sched = ProactiveScheduler(
            cfg=_make_cfg(score_weight_energy=0.0),
            presence=_make_presence(
                datetime.now(timezone.utc) - timedelta(hours=48)
            ),
        )
        self.assertEqual(sched.next_interval(), 4800)


class AdaptiveIntervalTest(unittest.TestCase):
    def test_high_score_short_interval(self) -> None:
        sched = ProactiveScheduler(
            cfg=_make_cfg(), presence=_make_presence(),
        )
        self.assertEqual(sched.next_interval(base_score=0.9), 2400)

    def test_low_score_long_interval(self) -> None:
        sched = ProactiveScheduler(
            cfg=_make_cfg(), presence=_make_presence(),
        )
        self.assertEqual(sched.next_interval(base_score=0.05), 4800)

    def test_jitter_keeps_interval_within_band(self) -> None:
        sched = ProactiveScheduler(
            cfg=_make_cfg(tick_jitter=0.3), presence=_make_presence(),
        )
        for _ in range(50):
            interval = sched.next_interval(base_score=0.5)
            # 2400 ± 30% = [1680, 3120]
            self.assertGreaterEqual(interval, 1680)
            self.assertLessEqual(interval, 3120)

    def test_custom_intervals(self) -> None:
        sched = ProactiveScheduler(
            cfg=_make_cfg(tick_interval_s1=100, tick_interval_s0=200),
            presence=_make_presence(),
        )
        self.assertEqual(sched.next_interval(base_score=0.5), 100)
        self.assertEqual(sched.next_interval(base_score=0.1), 200)


class TraceTest(unittest.TestCase):
    def test_trace_records_three_fields(self) -> None:
        traces, trace = _collect_traces()
        sched = ProactiveScheduler(
            cfg=_make_cfg(), presence=_make_presence(), trace_fn=trace,
        )
        sched.next_interval(base_score=0.5)
        entry = traces[-1]
        self.assertIn("base_score", entry)
        self.assertIn("interval", entry)
        self.assertIn("mode", entry)
        self.assertEqual(entry["mode"], "adaptive")


class ScheduleModuleTest(unittest.IsolatedAsyncioTestCase):
    def _module(self, **sched_kwargs):
        defaults = dict(cfg=_make_cfg(), presence=_make_presence())
        defaults.update(sched_kwargs)
        return ProactiveScheduleModule(ProactiveScheduler(**defaults))

    async def test_module_writes_interval_slot(self) -> None:
        mod = self._module()
        frame = new_frame("s1")
        result = await mod.run(frame)
        self.assertIn(SLOT_NEXT_INTERVAL, result.slots)
        # 无 base_score + 从未活跃（presence 为 None 记录）
        # → energy=0 → 满分 0.40 → 短间隔档
        self.assertEqual(result.slots[SLOT_NEXT_INTERVAL], 2400)

    async def test_module_recent_user_long_interval(self) -> None:
        mod = self._module(
            presence=_make_presence(datetime.now(timezone.utc)),
        )
        frame = await mod.run(new_frame("s1"))
        # 用户刚活跃 → energy 高 → 分数低 → 长间隔
        self.assertEqual(frame.slots[SLOT_NEXT_INTERVAL], 4800)

    async def test_module_reads_base_score_slot(self) -> None:
        mod = self._module()
        frame = new_frame("s1", slots={SLOT_BASE_SCORE: 0.5})
        result = await mod.run(frame)
        # judge 给了高分 → 短间隔
        self.assertEqual(result.slots[SLOT_NEXT_INTERVAL], 2400)

    async def test_module_declares_slot_and_phase(self) -> None:
        mod = self._module()
        self.assertEqual(mod.slot, "schedule")
        self.assertEqual(mod.phase, "proactive.schedule")

    async def test_module_composes_with_runner(self) -> None:
        """挂进阶段管线与 judge 模块协同。"""
        from initiative.proactive_phases import ProactivePhaseRunner

        class FakeJudge:
            slot = "judge"
            phase = "proactive.judge"

            async def run(self, frame):
                frame.slots[SLOT_BASE_SCORE] = 0.9
                return frame

        runner = ProactivePhaseRunner([FakeJudge(), self._module()])
        frame = await runner.run(new_frame("s1"))
        # judge 先写高分 → schedule 读到 → 短间隔
        self.assertEqual(frame.slots[SLOT_NEXT_INTERVAL], 2400)


if __name__ == "__main__":
    unittest.main()
