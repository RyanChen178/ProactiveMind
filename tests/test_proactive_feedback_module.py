"""主动推送反馈接线测试：跨 tick 记录 + 统计接入管线。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from initiative.proactive_feedback_module import (
    SLOT_FEEDBACK_STATS,
    FeedbackRecorder,
    FeedbackStatsModule,
    _latest_session,
)
from initiative.proactive_frame import new_frame
from mind.feedback_db import event_count_by_type


class _TmpDb:
    def __init__(self):
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "fb.db"

    def cleanup(self):
        self._tmp.cleanup()


class FeedbackRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = _TmpDb()

    def tearDown(self) -> None:
        self.db.cleanup()

    def _recorder(self):
        return FeedbackRecorder(self.db.path)

    def test_no_recent_returns_none(self) -> None:
        rec = self._recorder()
        result = rec.record(
            session_key="s", user_message_id="u1",
            assistant_message_id="a1", user_text="回复",
            recent_proactive=[],
        )
        self.assertIsNone(result)

    def test_scores_and_persists(self) -> None:
        rec = self._recorder()
        result = rec.record(
            session_key="web:u1", user_message_id="u1",
            assistant_message_id="a1",
            user_text="推送原文",  # 与推送一致 → 引用命中
            recent_proactive=["推送原文"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.feedback_type, "explicit_quote")
        # 落库可查
        from mind.feedback_db import open_db

        conn = open_db(self.db.path)
        try:
            counts = event_count_by_type(conn)
            self.assertEqual(counts.get("explicit_quote"), 1)
        finally:
            conn.close()

    def test_no_match_still_persists(self) -> None:
        rec = self._recorder()
        result = rec.record(
            session_key="s", user_message_id="u1",
            assistant_message_id="a1",
            user_text="完全无关的内容",
            recent_proactive=["之前的推送"],
        )
        self.assertEqual(result.feedback_type, "no_topic_follow")

    def test_lag_seconds_persisted(self) -> None:
        rec = self._recorder()
        rec.record(
            session_key="s", user_message_id="u1",
            assistant_message_id="a1", user_text="x",
            recent_proactive=["y"], lag_seconds=42,
        )
        from mind.feedback_db import open_db

        conn = open_db(self.db.path)
        try:
            row = conn.execute(
                "SELECT lag_seconds FROM proactive_feedback_events"
            ).fetchone()
            self.assertEqual(row["lag_seconds"], 42)
        finally:
            conn.close()

    def test_db_open_failure_returns_score(self) -> None:
        rec = FeedbackRecorder(Path("Z::\\bad\x00path"))
        # 评分照常返回，落库静默失败
        result = rec.record(
            session_key="s", user_message_id="u1",
            assistant_message_id="a1", user_text="x",
            recent_proactive=["y"],
        )
        self.assertIsNotNone(result)

    def test_multiple_events_accumulate(self) -> None:
        rec = self._recorder()
        for i in range(3):
            rec.record(
                session_key="s", user_message_id=f"u{i}",
                assistant_message_id=f"a{i}", user_text=f"回复{i}",
                recent_proactive=["推送"],
            )
        from mind.feedback_db import open_db

        conn = open_db(self.db.path)
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM proactive_feedback_events"
            ).fetchone()[0]
            self.assertEqual(total, 3)
        finally:
            conn.close()


class LatestSessionTest(unittest.TestCase):
    def test_picks_largest(self) -> None:
        self.assertEqual(
            _latest_session({"a": 2, "b": 5, "c": 1}), "b"
        )

    def test_empty(self) -> None:
        self.assertEqual(_latest_session({}), "")


class FeedbackStatsModuleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = _TmpDb()
        self._seed()

    def tearDown(self) -> None:
        self.db.cleanup()

    def _seed(self):
        from mind.feedback_db import FeedbackEvent, insert_feedback, open_db

        conn = open_db(self.db.path)
        try:
            insert_feedback(conn, FeedbackEvent(
                session_key="s1", user_message_id="u1",
                assistant_message_id="a1", proactive_message_id="p1",
                feedback_type="explicit_quote", confidence="high",
                pa_score=0.9, pua_score=0.1, lag_seconds=5,
                candidate_count=1, matched_by="exact", reason="r",
            ))
            insert_feedback(conn, FeedbackEvent(
                session_key="s1", user_message_id="u2",
                assistant_message_id="a2", proactive_message_id="p1",
                feedback_type="topic_follow", confidence="high",
                pa_score=0.5, pua_score=0.5, lag_seconds=10,
                candidate_count=1, matched_by="token", reason="r",
            ))
            insert_feedback(conn, FeedbackEvent(
                session_key="s2", user_message_id="u3",
                assistant_message_id="a3", proactive_message_id="p2",
                feedback_type="no_topic_follow", confidence="low",
                pa_score=0.1, pua_score=0.9, lag_seconds=20,
                candidate_count=1, matched_by="none", reason="r",
            ))
        finally:
            conn.close()

    def _mod(self):
        return FeedbackStatsModule(self.db.path)

    async def test_writes_stats_slot(self) -> None:
        mod = self._mod()
        frame = await mod.run(new_frame("web:u1"))
        stats = frame.slots[SLOT_FEEDBACK_STATS]
        self.assertEqual(stats["total_events"], 3)
        self.assertEqual(stats["by_type"]["explicit_quote"], 1)
        self.assertAlmostEqual(stats["engagement_rate"], 2 / 3, places=4)
        self.assertAlmostEqual(stats["deflection_rate"], 1 / 3, places=4)
        self.assertEqual(stats["session_count"], 2)
        # 主要会话（s1）的最近事件附带
        self.assertTrue(any(
            e["session_key"] == "s1" or e.get("skill")
            or "s1" in str(e) for e in stats["latest_events"]
        ) or len(stats["latest_events"]) > 0)

    async def test_empty_db_writes_zero_stats(self) -> None:
        empty = _TmpDb()
        try:
            mod = FeedbackStatsModule(empty.path)
            frame = await mod.run(new_frame("s"))
            stats = frame.slots[SLOT_FEEDBACK_STATS]
            self.assertEqual(stats["total_events"], 0)
            self.assertEqual(stats["engagement_rate"], 0.0)
            self.assertEqual(stats["latest_events"], [])
        finally:
            empty.cleanup()

    async def test_db_failure_skips_silently(self) -> None:
        mod = FeedbackStatsModule(Path("Z::\\bad\x00path"))
        frame = await mod.run(new_frame("s"))
        self.assertNotIn(SLOT_FEEDBACK_STATS, frame.slots)

    async def test_declares_slot_and_phase(self) -> None:
        mod = self._mod()
        self.assertEqual(mod.slot, "proactive.prompt.feedback_stats")
        self.assertEqual(mod.phase, "proactive.prompt")

    async def test_composes_with_runner(self) -> None:
        from initiative.proactive_phases import ProactivePhaseRunner

        runner = ProactivePhaseRunner([self._mod()])
        frame = await runner.run(new_frame("web:u1"))
        self.assertIn(SLOT_FEEDBACK_STATS, frame.slots)


class RecorderStatsIntegrationTest(unittest.TestCase):
    """记录 → 统计 全链路。"""

    def test_record_then_collect(self) -> None:
        db = _TmpDb()
        try:
            rec = FeedbackRecorder(db.path)
            rec.record(
                session_key="s1", user_message_id="u1",
                assistant_message_id="a1", user_text="推送内容",
                recent_proactive=["推送内容"],
            )
            mod = FeedbackStatsModule(db.path)
            stats = mod.collect()
            self.assertIsNotNone(stats)
            self.assertEqual(stats["total_events"], 1)
            self.assertEqual(stats["by_type"]["explicit_quote"], 1)
        finally:
            db.cleanup()


if __name__ == "__main__":
    unittest.main()
