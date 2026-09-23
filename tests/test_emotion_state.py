"""VAD 情绪状态测试：持久化、衰减、反馈增量、tick effect。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mind.emotion_state import (
    DECAY_HALF_LIFE_HOURS,
    EmotionState,
    apply_feedback,
    classify_feedback_delta,
    compute_tick_effect,
    get_state,
    open_db,
)


class _DbMixin:
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.conn = open_db(Path(self._tmp.name) / "emotion.db")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()


class OpenDbTest(_DbMixin, unittest.TestCase):
    def test_initial_state_is_neutral(self) -> None:
        state = get_state(self.conn)
        self.assertEqual(state.valence, 0.0)
        self.assertEqual(state.arousal, 0.0)
        self.assertEqual(state.dominance, 0.0)

    def test_idempotent_open(self) -> None:
        self.conn.close()
        self.conn = open_db(Path(self._tmp.name) / "emotion.db")
        self.assertEqual(get_state(self.conn).valence, 0.0)


class FeedbackDeltaTest(unittest.TestCase):
    def test_explicit_quote(self) -> None:
        v, d, reason = classify_feedback_delta("explicit_quote")
        self.assertEqual(reason, "explicit_quote")
        self.assertGreater(v, 0)
        self.assertGreater(d, 0)

    def test_topic_follow_high(self) -> None:
        _, _, reason = classify_feedback_delta("topic_follow", confidence="high")
        self.assertEqual(reason, "topic_follow_high")

    def test_topic_follow_medium(self) -> None:
        _, _, reason = classify_feedback_delta("topic_follow")
        self.assertEqual(reason, "topic_follow_medium")

    def test_no_topic_follow_negative(self) -> None:
        v, d, reason = classify_feedback_delta("no_topic_follow")
        self.assertEqual(reason, "no_topic_follow")
        self.assertLess(v, 0)
        self.assertLess(d, 0)

    def test_unknown_falls_back_to_neutral(self) -> None:
        v, d, reason = classify_feedback_delta("totally_unknown_type")
        self.assertEqual(reason, "neutral_feedback")
        self.assertEqual(v, 0.0)
        self.assertEqual(d, 0.0)


class ApplyFeedbackTest(_DbMixin, unittest.TestCase):
    def test_explicit_quote_increases_valence(self) -> None:
        after = apply_feedback(
            self.conn,
            source_event_id="evt-1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="explicit_quote",
        )
        self.assertGreater(after.valence, 0.0)
        self.assertGreater(after.dominance, 0.0)

    def test_no_topic_follow_decreases_dominance(self) -> None:
        apply_feedback(
            self.conn,
            source_event_id="evt-1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="explicit_quote",
        )
        first_after = get_state(self.conn)
        apply_feedback(
            self.conn,
            source_event_id="evt-2",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="no_topic_follow",
        )
        second_after = get_state(self.conn)
        self.assertLess(second_after.dominance, first_after.dominance)

    def test_clamps_within_bounds(self) -> None:
        for i in range(50):
            apply_feedback(
                self.conn,
                source_event_id=f"evt-{i}",
                source_plugin="emotion",
                session_key="s1",
                feedback_type="explicit_quote",
            )
        state = get_state(self.conn)
        self.assertLessEqual(state.valence, 1.0)
        self.assertGreaterEqual(state.valence, -1.0)
        self.assertLessEqual(state.dominance, 1.0)

    def test_event_audit_idempotent_on_same_event_id(self) -> None:
        apply_feedback(
            self.conn,
            source_event_id="evt-1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="explicit_quote",
        )
        first = get_state(self.conn)
        apply_feedback(
            self.conn,
            source_event_id="evt-1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="explicit_quote",
        )
        second = get_state(self.conn)
        self.assertGreater(second.valence, first.valence)
        count = self.conn.execute(
            "SELECT COUNT(*) AS n FROM emotion_events WHERE source_event_id = ?",
            ("evt-1",),
        ).fetchone()["n"]
        self.assertEqual(count, 1)

    def test_event_audit_recorded(self) -> None:
        apply_feedback(
            self.conn,
            source_event_id="evt-1",
            source_plugin="emotion",
            session_key="sess-x",
            feedback_type="explicit_quote",
            payload={"detail": "x"},
        )
        row = self.conn.execute(
            "SELECT source_plugin, session_key, reason, payload_json "
            "FROM emotion_events WHERE source_event_id = ?",
            ("evt-1",),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["source_plugin"], "emotion")
        self.assertEqual(row["session_key"], "sess-x")
        self.assertEqual(row["reason"], "explicit_quote")
        self.assertIn('"detail"', row["payload_json"])


class DecayTest(_DbMixin, unittest.TestCase):
    def _seed_state(self, hours_ago: float, valence: float = 0.8):
        old_iso = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        self.conn.execute(
            "UPDATE emotion_state SET valence = ?, arousal = ?, dominance = ?, updated_at = ? WHERE id = 1",
            (valence, 0.6, 0.4, old_iso),
        )
        self.conn.commit()

    def test_decay_recent_state(self) -> None:
        self._seed_state(hours_ago=10 / 60)
        after = apply_feedback(
            self.conn,
            source_event_id="e1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="neutral",
        )
        self.assertAlmostEqual(after.valence, 0.8, delta=0.05)

    def test_decay_old_state(self) -> None:
        self._seed_state(hours_ago=DECAY_HALF_LIFE_HOURS)
        after = apply_feedback(
            self.conn,
            source_event_id="e1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="neutral",
        )
        self.assertAlmostEqual(after.valence, 0.4, delta=0.05)

    def test_decay_very_old_state(self) -> None:
        self._seed_state(hours_ago=DECAY_HALF_LIFE_HOURS * 5)
        after = apply_feedback(
            self.conn,
            source_event_id="e1",
            source_plugin="emotion",
            session_key="s1",
            feedback_type="neutral",
        )
        self.assertLess(abs(after.valence), 0.05)


class ComputeTickEffectTest(_DbMixin, unittest.TestCase):
    def test_neutral_state_keeps_threshold(self) -> None:
        effect = compute_tick_effect(
            self.conn, tick_id="t1", session_key="s1", base_threshold=0.6,
        )
        self.assertEqual(effect.tone_label, "neutral")
        self.assertEqual(effect.threshold_delta, 0.0)
        self.assertEqual(effect.final_threshold, 0.6)
        self.assertEqual(effect.expected_effect, "tone_neutral")

    def test_positive_state_lowers_threshold(self) -> None:
        for i in range(5):
            apply_feedback(
                self.conn,
                source_event_id=f"e{i}",
                source_plugin="emotion",
                session_key="s1",
                feedback_type="explicit_quote",
            )
        effect = compute_tick_effect(
            self.conn, tick_id="t1", session_key="s1", base_threshold=0.6,
        )
        self.assertEqual(effect.tone_label, "positive")
        self.assertLess(effect.final_threshold, 0.6)
        self.assertEqual(effect.expected_effect, "tone_warm")

    def test_negative_state_raises_threshold(self) -> None:
        for i in range(10):
            apply_feedback(
                self.conn,
                source_event_id=f"e{i}",
                source_plugin="emotion",
                session_key="s1",
                feedback_type="no_topic_follow",
            )
        effect = compute_tick_effect(
            self.conn, tick_id="t1", session_key="s1", base_threshold=0.6,
        )
        self.assertEqual(effect.tone_label, "negative")
        self.assertGreater(effect.final_threshold, 0.6)
        self.assertEqual(effect.expected_effect, "tone_cautious")

    def test_effect_recorded_in_db(self) -> None:
        compute_tick_effect(
            self.conn, tick_id="t-unique", session_key="s1", base_threshold=0.5,
        )
        row = self.conn.execute(
            "SELECT tone_label, base_threshold FROM emotion_effects WHERE tick_id = ?",
            ("t-unique",),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["tone_label"], "neutral")
        self.assertEqual(row["base_threshold"], 0.5)


class EmotionEffectDataclassTest(unittest.TestCase):
    def test_as_dict_round_trip(self) -> None:
        state = EmotionState(0.5, -0.2, 0.1, "2026-01-01T00:00:00+00:00")
        d = state.as_dict()
        self.assertEqual(d["valence"], 0.5)
        self.assertEqual(d["updated_at"], "2026-01-01T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()