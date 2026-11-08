"""昼夜 gate 与情绪 prompt 模块测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from initiative.proactive_daynight_module import (
    SLOT_EFFECT as DN_EFFECT,
    SLOT_GATE_REASON,
    SLOT_PASS_PROBABILITY,
    DayNightConfig,
    DayNightGateModule,
    _in_window,
    _parse_hhmm,
)
from initiative.proactive_emotion_module import (
    SLOT_BASE_THRESHOLD,
    SLOT_EFFECT as EM_EFFECT,
    SLOT_PROMPT_SECTION,
    EmotionPromptModule,
)
from initiative.proactive_frame import new_frame
from mind.emotion_state import apply_feedback, open_db


def _frame_at(hour_utc: int, minute: int = 0):
    """构造 started_at 固定在指定 UTC 时刻的 frame。"""
    frame = new_frame("web:u1")
    # frozen dataclass → 重建
    from dataclasses import replace

    started = datetime(2026, 6, 1, hour_utc, minute, tzinfo=timezone.utc)
    frame.input = replace(frame.input, started_at=started)
    return frame


class ParseHhmmTest(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(_parse_hhmm("06:00").hour, 6)
        self.assertEqual(_parse_hhmm("23:59").minute, 59)

    def test_invalid_format(self) -> None:
        for bad in ("0600", "ab:cd", "", "25:00", "12:60", "12:00:30"):
            with self.assertRaises(ValueError):
                _parse_hhmm(bad)


class InWindowTest(unittest.TestCase):
    def test_same_day_window(self) -> None:
        # 00:00-06:00
        from datetime import time as t

        self.assertTrue(_in_window(t(3, 0), t(0, 0), t(6, 0)))
        self.assertFalse(_in_window(t(12, 0), t(0, 0), t(6, 0)))
        self.assertFalse(_in_window(t(6, 0), t(0, 0), t(6, 0)))  # 右开

    def test_cross_midnight_window(self) -> None:
        from datetime import time as t

        self.assertTrue(_in_window(t(23, 0), t(22, 0), t(6, 0)))
        self.assertTrue(_in_window(t(2, 0), t(22, 0), t(6, 0)))
        self.assertFalse(_in_window(t(12, 0), t(22, 0), t(6, 0)))

    def test_zero_length_window(self) -> None:
        from datetime import time as t

        self.assertFalse(_in_window(t(0, 0), t(6, 0), t(6, 0)))


class DayNightConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = DayNightConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.timezone, "Asia/Shanghai")
        self.assertEqual(cfg.pass_probability, 0.15)

    def test_probability_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            DayNightConfig(pass_probability=1.5)

    def test_bad_time_format(self) -> None:
        with self.assertRaises(ValueError):
            DayNightConfig(start="0600")

    def test_bad_timezone(self) -> None:
        with self.assertRaises(ValueError):
            DayNightConfig(timezone="Not/AZone")


class DayNightModuleTest(unittest.IsolatedAsyncioTestCase):
    def _mod(self, **cfg_kwargs):
        return DayNightGateModule(DayNightConfig(**cfg_kwargs))

    async def test_daytime_untouched(self) -> None:
        mod = self._mod(start="00:00", end="06:00")
        frame = _frame_at(12)  # UTC 12:00 = 上海 20:00（白天）
        result = await mod.run(frame)
        self.assertNotIn(SLOT_PASS_PROBABILITY, result.slots)
        self.assertNotIn(SLOT_GATE_REASON, result.slots)

    async def test_night_slots_written(self) -> None:
        mod = self._mod(start="00:00", end="06:00")
        frame = _frame_at(19)  # UTC 19:00 = 上海 03:00（夜间）
        result = await mod.run(frame)
        self.assertEqual(
            result.slots[SLOT_PASS_PROBABILITY], 0.15
        )
        self.assertEqual(result.slots[SLOT_GATE_REASON], "quiet_hours")
        effect = result.slots[DN_EFFECT]
        self.assertEqual(effect["provider_name"], "daynight_gate")
        self.assertEqual(effect["start"], "00:00")

    async def test_cross_midnight_night(self) -> None:
        mod = self._mod(start="22:00", end="06:00")
        frame = _frame_at(16)  # UTC 16:00 = 上海 00:00（跨午夜窗口内）
        result = await mod.run(frame)
        self.assertIn(SLOT_PASS_PROBABILITY, result.slots)

    async def test_probability_takes_min_with_existing(self) -> None:
        mod = self._mod(pass_probability=0.15)
        frame = _frame_at(19)
        frame.slots[SLOT_PASS_PROBABILITY] = 0.05  # 已有更严格值
        result = await mod.run(frame)
        self.assertEqual(result.slots[SLOT_PASS_PROBABILITY], 0.05)

    async def test_probability_min_other_direction(self) -> None:
        mod = self._mod(pass_probability=0.10)
        frame = _frame_at(19)
        frame.slots[SLOT_PASS_PROBABILITY] = 0.50  # 已有更宽松值
        result = await mod.run(frame)
        self.assertEqual(result.slots[SLOT_PASS_PROBABILITY], 0.10)

    async def test_disabled_untouched(self) -> None:
        mod = self._mod(enabled=False)
        frame = _frame_at(19)  # 时段在夜间但被禁用
        result = await mod.run(frame)
        self.assertNotIn(SLOT_PASS_PROBABILITY, result.slots)

    async def test_declares_slot_and_phase(self) -> None:
        mod = self._mod()
        self.assertEqual(mod.slot, "proactive.gate.daynight")
        self.assertEqual(mod.phase, "proactive.gate")

    async def test_composes_with_runner(self) -> None:
        from initiative.proactive_phases import ProactivePhaseRunner

        mod = self._mod()
        runner = ProactivePhaseRunner([mod])
        frame = await runner.run(_frame_at(19))
        self.assertIn(SLOT_PASS_PROBABILITY, frame.slots)


class EmotionModuleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "emotion" / "emotion.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _mod(self):
        return EmotionPromptModule(self.db_path)

    async def test_neutral_state_writes_slots(self) -> None:
        open_db(self.db_path).close()  # 初始化为中性
        mod = self._mod()
        frame = await mod.run(new_frame("web:u1"))
        self.assertIn(SLOT_PROMPT_SECTION, frame.slots)
        self.assertIn("中性", frame.slots[SLOT_PROMPT_SECTION])
        effect = frame.slots[EM_EFFECT]
        self.assertEqual(effect["provider_name"], "emotion")
        self.assertEqual(effect["tone_label"], "neutral")
        self.assertEqual(effect["base_threshold"], 0.60)

    async def test_positive_state_adjusts_threshold_down(self) -> None:
        conn = open_db(self.db_path)
        try:
            for i in range(8):
                apply_feedback(
                    conn, source_event_id=f"e{i}", source_plugin="t",
                    session_key="s", feedback_type="explicit_quote",
                )
        finally:
            conn.close()
        mod = self._mod()
        frame = await mod.run(new_frame("web:u1"))
        effect = frame.slots[EM_EFFECT]
        self.assertEqual(effect["tone_label"], "positive")
        self.assertLess(effect["final_threshold"], effect["base_threshold"])
        self.assertIn("积极", frame.slots[SLOT_PROMPT_SECTION])

    async def test_base_threshold_read_from_slots(self) -> None:
        open_db(self.db_path).close()
        mod = self._mod()
        frame = new_frame("s", slots={SLOT_BASE_THRESHOLD: 0.75})
        result = await mod.run(frame)
        self.assertEqual(result.slots[EM_EFFECT]["base_threshold"], 0.75)

    async def test_base_threshold_out_of_range_clamped(self) -> None:
        open_db(self.db_path).close()
        mod = self._mod()
        frame = new_frame("s", slots={SLOT_BASE_THRESHOLD: 2.0})
        result = await mod.run(frame)
        self.assertEqual(result.slots[EM_EFFECT]["base_threshold"], 1.0)

    async def test_missing_db_skips_silently(self) -> None:
        mod = EmotionPromptModule(Path(self._tmp.name) / "nope" / "x.db")
        # 目录不存在也能 open_db（自动 mkdir），所以这里用坏路径注入异常
        mod._db_path = "Z::\\invalid\x00path"
        frame = await mod.run(new_frame("s"))
        self.assertNotIn(SLOT_PROMPT_SECTION, frame.slots)

    async def test_declares_slot_and_phase(self) -> None:
        mod = self._mod()
        self.assertEqual(mod.slot, "proactive.prompt.emotion")
        self.assertEqual(mod.phase, "proactive.prompt")

    async def test_composes_with_runner(self) -> None:
        from initiative.proactive_phases import ProactivePhaseRunner

        open_db(self.db_path).close()
        runner = ProactivePhaseRunner([self._mod()])
        frame = await runner.run(new_frame("web:u1"))
        self.assertIn(SLOT_PROMPT_SECTION, frame.slots)


class BothModulesTogetherTest(unittest.IsolatedAsyncioTestCase):
    """昼夜 gate + 情绪 prompt 同时挂进管线。"""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "emotion.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_two_plugins_coexist(self) -> None:
        from initiative.proactive_phases import ProactivePhaseRunner

        open_db(self.db_path).close()
        runner = ProactivePhaseRunner([
            DayNightGateModule(DayNightConfig()),
            EmotionPromptModule(self.db_path),
        ])
        frame = await runner.run(_frame_at(19))  # 上海 03:00 夜间
        self.assertIn(SLOT_PASS_PROBABILITY, frame.slots)  # gate 生效
        self.assertIn(SLOT_PROMPT_SECTION, frame.slots)    # 情绪生效


if __name__ == "__main__":
    unittest.main()
