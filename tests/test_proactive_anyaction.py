"""AnyAction 频控测试。"""

from __future__ import annotations

import random
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

from initiative.proactive_anyaction import (
    AnyActionGate,
    DEFAULT_QUOTA_FILE,
    QuotaSnapshot,
    QuotaStore,
)


def _make_cfg(**overrides):
    base = {
        "anyaction_daily_max_actions": 5,
        "anyaction_min_interval_seconds": 60,
        "anyaction_idle_scale_minutes": 15.0,
        "anyaction_probability_min": 0.05,
        "anyaction_probability_max": 0.50,
        "anyaction_reset_hour_local": 12,
        "anyaction_timezone": "UTC",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class QuotaStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / DEFAULT_QUOTA_FILE

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_initial(self):
        store = QuotaStore(self.path)
        snap = store.snapshot(
            now_utc=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
            reset_hour=12, timezone_name="UTC",
        )
        self.assertEqual(snap.used, 0)
        self.assertIsNone(snap.last_action_at)

    def test_record_increments_used(self):
        store = QuotaStore(self.path)
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        store.record_action(
            now_utc=now, reset_hour=12, timezone_name="UTC"
        )
        snap = store.snapshot(
            now_utc=now, reset_hour=12, timezone_name="UTC"
        )
        self.assertEqual(snap.used, 1)
        self.assertEqual(snap.last_action_at, now)

    def test_rollover_resets_used(self):
        store = QuotaStore(self.path)
        day1 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        store.record_action(
            now_utc=day1, reset_hour=12, timezone_name="UTC"
        )
        day2 = datetime(2026, 1, 2, 12, 30, tzinfo=timezone.utc)
        snap = store.snapshot(
            now_utc=day2, reset_hour=12, timezone_name="UTC"
        )
        self.assertEqual(snap.used, 0)
        self.assertNotEqual(snap.window_key, "")

    def test_persists_across_instances(self):
        store1 = QuotaStore(self.path)
        store1.record_action(
            now_utc=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
            reset_hour=12, timezone_name="UTC",
        )
        store2 = QuotaStore(self.path)
        snap = store2.snapshot(
            now_utc=datetime(2026, 1, 1, 13, tzinfo=timezone.utc),
            reset_hour=12, timezone_name="UTC",
        )
        self.assertEqual(snap.used, 1)

    def test_corrupted_file_falls_back_to_empty(self):
        self.path.write_text("not json{", encoding="utf-8")
        store = QuotaStore(self.path)
        snap = store.snapshot(
            now_utc=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
            reset_hour=12, timezone_name="UTC",
        )
        self.assertEqual(snap.used, 0)


class AnyActionGateQuotaTest(unittest.TestCase):
    def _quota(self, used=0, last_action_at=None):
        snap = QuotaSnapshot(
            window_key="w1", next_reset_at=datetime.now(timezone.utc),
            used=used, last_action_at=last_action_at,
        )
        m = MagicMock()
        m.snapshot = MagicMock(return_value=snap)
        m.record_action = MagicMock()
        return m

    def test_quota_exhausted_blocks(self):
        quota = self._quota(used=5)
        cfg = _make_cfg(anyaction_daily_max_actions=5)
        gate = AnyActionGate(cfg=cfg, quota_store=quota, rng=random.Random(0))
        should, meta = gate.should_act(
            now_utc=datetime.now(timezone.utc), last_user_at=None,
        )
        self.assertFalse(should)
        self.assertEqual(meta["reason"], "quota_exhausted")

    def test_min_interval_blocks(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        quota = self._quota(used=1, last_action_at=now - timedelta(seconds=10))
        cfg = _make_cfg(anyaction_min_interval_seconds=60)
        gate = AnyActionGate(cfg=cfg, quota_store=quota, rng=random.Random(0))
        should, meta = gate.should_act(now_utc=now, last_user_at=None)
        self.assertFalse(should)
        self.assertEqual(meta["reason"], "min_interval")

    def test_min_interval_satisfied(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        quota = self._quota(used=1, last_action_at=now - timedelta(seconds=120))
        cfg = _make_cfg(anyaction_min_interval_seconds=60)
        gate = AnyActionGate(cfg=cfg, quota_store=quota, rng=random.Random(0))
        gate.should_act(now_utc=now, last_user_at=None)


class AnyActionGateProbabilityTest(unittest.TestCase):
    def _gate(self, cfg, rng_seed=0):
        snap = QuotaSnapshot(
            window_key="w1",
            next_reset_at=datetime.now(timezone.utc),
            used=0, last_action_at=None,
        )
        q = MagicMock()
        q.snapshot = MagicMock(return_value=snap)
        return AnyActionGate(cfg=cfg, quota_store=q, rng=random.Random(rng_seed))

    def test_idle_long_increases_p(self):
        cfg = _make_cfg(
            anyaction_idle_scale_minutes=15.0,
            anyaction_probability_min=0.05,
            anyaction_probability_max=0.95,
        )
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        gate = self._gate(cfg)
        _, meta1 = gate.should_act(now_utc=now, last_user_at=now)
        _, meta2 = gate.should_act(
            now_utc=now,
            last_user_at=now - timedelta(hours=2),
        )
        self.assertGreater(meta2["p_act"], meta1["p_act"])

    def test_no_last_user_uses_2x_idle_scale(self):
        cfg = _make_cfg(
            anyaction_idle_scale_minutes=15.0,
            anyaction_probability_min=0.0,
            anyaction_probability_max=1.0,
        )
        gate = self._gate(cfg)
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        _, meta = gate.should_act(now_utc=now, last_user_at=None)
        self.assertAlmostEqual(meta["p_act"], 1 - 2.71828 ** -2, places=2)

    def test_fixed_rng_zero_acts(self):
        cfg = _make_cfg(
            anyaction_probability_min=0.0, anyaction_probability_max=1.0,
        )
        fixed_rng = MagicMock()
        fixed_rng.random = MagicMock(return_value=0.0)
        snap = QuotaSnapshot(
            window_key="w1",
            next_reset_at=datetime.now(timezone.utc),
            used=0, last_action_at=None,
        )
        q = MagicMock()
        q.snapshot = MagicMock(return_value=snap)
        gate = AnyActionGate(cfg=cfg, quota_store=q, rng=fixed_rng)
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        should, meta = gate.should_act(now_utc=now, last_user_at=None)
        self.assertTrue(should)
        self.assertEqual(meta["draw"], 0.0)

    def test_fixed_rng_high_blocks(self):
        cfg = _make_cfg(
            anyaction_probability_min=0.0, anyaction_probability_max=1.0,
        )
        fixed_rng = MagicMock()
        fixed_rng.random = MagicMock(return_value=0.99)
        snap = QuotaSnapshot(
            window_key="w1",
            next_reset_at=datetime.now(timezone.utc),
            used=0, last_action_at=None,
        )
        q = MagicMock()
        q.snapshot = MagicMock(return_value=snap)
        gate = AnyActionGate(cfg=cfg, quota_store=q, rng=fixed_rng)
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        should, meta = gate.should_act(now_utc=now, last_user_at=None)
        self.assertFalse(should)
        self.assertEqual(meta["draw"], 0.99)


class AnyActionRecordTest(unittest.TestCase):
    def test_record_delegates_to_quota(self):
        q = MagicMock()
        gate = AnyActionGate(
            cfg=_make_cfg(),
            quota_store=q,
            rng=random.Random(0),
        )
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        gate.record_action(now_utc=now)
        q.record_action.assert_called_once()


if __name__ == "__main__":
    unittest.main()
