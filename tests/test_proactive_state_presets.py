"""主动链路 State + Presets 单元测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from initiative.proactive_presets import (
    ALLOWED_OVERRIDE_KEYS,
    PRESETS,
    STRATEGY_PARAMS,
    get_preset,
    preset_names,
    resolve_preset,
)
from initiative.proactive_state import (
    DeliveryRecord,
    ProactiveStateStore,
    _parse_iso,
    _utcnow,
)


class ProactiveStateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.store = ProactiveStateStore(Path(self._tmp.name) / "st.db")

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_delivery_record_new_insert_returns_true(self) -> None:
        rec = DeliveryRecord(
            session_key="s1",
            message_id="m1",
            pushed_at=_utcnow(),
        )
        self.assertTrue(self.store.record_delivery(rec))

    def test_delivery_record_duplicate_returns_false(self) -> None:
        rec = DeliveryRecord(
            session_key="s1", message_id="m1", pushed_at=_utcnow()
        )
        self.assertTrue(self.store.record_delivery(rec))
        self.assertFalse(self.store.record_delivery(rec))

    def test_recent_delivery_messages(self) -> None:
        ts = _utcnow()
        for i in range(3):
            self.store.record_delivery(DeliveryRecord(
                session_key="s1", message_id=f"m{i}", pushed_at=ts
            ))
        msgs = self.store.recent_delivery_messages("s1", limit=10)
        self.assertEqual(set(msgs), {"m0", "m1", "m2"})

    def test_last_delivery_at(self) -> None:
        ts = _utcnow()
        self.store.record_delivery(DeliveryRecord(
            session_key="s1", message_id="m1", pushed_at=ts
        ))
        result = self.store.last_delivery_at("s1")
        self.assertIsNotNone(result)
        self.assertEqual(result.replace(tzinfo=timezone.utc), ts)

    def test_last_delivery_at_unknown_session(self) -> None:
        self.assertIsNone(self.store.last_delivery_at("nope"))

    def test_record_context_only_idempotent(self) -> None:
        ts = _utcnow()
        self.store.record_context_only("s1", "morning", ts)
        self.store.record_context_only("s1", "morning", ts)  # 覆盖
        result = self.store.last_context_only_at("s1", "morning")
        self.assertEqual(result.replace(tzinfo=timezone.utc), ts)

    def test_last_context_only_at_unknown(self) -> None:
        self.assertIsNone(
            self.store.last_context_only_at("s1", "no_such_kind")
        )

    def test_log_tick_inserts_record(self) -> None:
        self.store.log_tick(
            "s1", "executed",
            base_score=0.5,
            pushed_message_id="m1",
            note={"reason": "test"},
        )
        rows = self.store.recent_ticks("s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "executed")
        self.assertEqual(rows[0]["base_score"], 0.5)

    def test_close_is_idempotent(self) -> None:
        self.store.close()
        self.store.close()  # 不抛错

    def test_different_sessions_isolated(self) -> None:
        ts = _utcnow()
        self.store.record_delivery(DeliveryRecord(
            session_key="s1", message_id="m1", pushed_at=ts
        ))
        self.assertEqual(self.store.last_delivery_at("s2"), None)


class ParseIsoTest(unittest.TestCase):
    def test_valid_iso(self) -> None:
        dt = _parse_iso("2026-01-01T12:00:00+00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_parse_iso("not a date"))


class GetPresetTest(unittest.TestCase):
    def test_known_name(self) -> None:
        p = get_preset("daily")
        self.assertIn("trigger", p)
        self.assertEqual(p["gate"]["judge_send_threshold"], 0.60)

    def test_unknown_falls_back_to_daily(self) -> None:
        p = get_preset("does_not_exist")
        self.assertEqual(p, get_preset("daily"))


class PresetNamesTest(unittest.TestCase):
    def test_contains_daily_and_quiet(self) -> None:
        names = preset_names()
        self.assertIn("daily", names)
        self.assertIn("quiet", names)

    def test_preserves_declaration_order(self) -> None:
        names = preset_names()
        self.assertEqual(names.index("daily"), 0)
        self.assertEqual(names.index("quiet"), 1)


class ResolvePresetTest(unittest.TestCase):
    def test_no_overrides_returns_base(self) -> None:
        p = resolve_preset("quiet")
        self.assertEqual(p, get_preset("quiet"))

    def test_allowed_override_applied(self) -> None:
        p = resolve_preset(
            "daily",
            overrides={"gate": {"judge_send_threshold": 0.95}},
        )
        self.assertEqual(p["gate"]["judge_send_threshold"], 0.95)

    def test_disallowed_override_silently_dropped(self) -> None:
        # "score_weight_energy" 不在白名单里
        p = resolve_preset(
            "daily",
            overrides={"trigger": {"score_weight_energy": 999}},
        )
        self.assertNotIn("score_weight_energy", p["trigger"])

    def test_unknown_section_ignored(self) -> None:
        p = resolve_preset(
            "daily",
            overrides={"nonexistent_section": {"foo": 1}},
        )
        self.assertNotIn("nonexistent_section", p)

    def test_allowed_keys_match_preset_structure(self) -> None:
        """白名单 key 必须都在对应预设中存在。"""
        for name, preset in PRESETS.items():
            for section, allowed_keys in ALLOWED_OVERRIDE_KEYS.items():
                preset_section = preset[section]  # type: ignore[index]
                for key in allowed_keys:
                    self.assertIn(
                        key, preset_section,
                        f"preset={name} section={section} key={key}",
                    )


class StrategyParamsTest(unittest.TestCase):
    def test_contains_expected_keys(self) -> None:
        self.assertIn("score_weight_energy", STRATEGY_PARAMS)
        self.assertIn("recent_chat_messages", STRATEGY_PARAMS)


if __name__ == "__main__":
    unittest.main()
