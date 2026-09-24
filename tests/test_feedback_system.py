"""反馈系统测试：scorer + db + stats 三件套。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mind.feedback_db import (
    FeedbackEvent,
    avg_metric_by_type,
    delete_session_events,
    event_count_by_type,
    events_for_proactive,
    insert_feedback,
    list_session_events,
    open_db,
)
from mind.feedback_scorer import (
    EXACT_MATCH_THRESHOLD,
    clean_text,
    is_proactive_message,
    normalize_quote_text,
    parse_quote_parts,
    score_feedback,
    tokenize,
)
from mind.feedback_stats import FeedbackReporter


class CleanTextTest(unittest.TestCase):
    def test_collapses_whitespace(self) -> None:
        self.assertEqual(clean_text("  a  b\nc\t"), "a b c")

    def test_truncates(self) -> None:
        self.assertEqual(len(clean_text("x" * 100, max_chars=10)), 10)

    def test_handles_empty(self) -> None:
        self.assertEqual(clean_text(""), "")


class NormalizeTest(unittest.TestCase):
    def test_strips_markdown(self) -> None:
        text = "Hello **World** _test_ [link](http://e.com)"
        out = normalize_quote_text(text)
        # _ 去除链接标签后应只剩 "hello world test link httpe.com" 一类规范化结果
        self.assertNotIn("**", out)
        self.assertNotIn("_", out)
        self.assertNotIn("[", out)

    def test_lowercase(self) -> None:
        self.assertEqual(normalize_quote_text("ABC"), "abc")


class ParseQuoteTest(unittest.TestCase):
    def test_with_marker(self) -> None:
        content = "被回复消息 m1：今天天气真好【你当前新消息】谢谢!"
        parts = parse_quote_parts(content)
        self.assertEqual(parts.quoted_text, "今天天气真好")
        self.assertEqual(parts.current_text, "谢谢!")

    def test_without_marker(self) -> None:
        content = "只是一条普通消息"
        parts = parse_quote_parts(content)
        self.assertIsNone(parts.quoted_text)
        self.assertEqual(parts.current_text, "只是一条普通消息")

    def test_empty(self) -> None:
        parts = parse_quote_parts("")
        self.assertIsNone(parts.quoted_text)
        self.assertEqual(parts.current_text, "")


class IsProactiveTest(unittest.TestCase):
    def test_dict_true(self) -> None:
        self.assertTrue(is_proactive_message({"proactive": True}))
        self.assertTrue(is_proactive_message({"is_proactive": "true"}))

    def test_dict_false(self) -> None:
        self.assertFalse(is_proactive_message({"proactive": False}))
        self.assertFalse(is_proactive_message({"foo": "bar"}))

    def test_string_format(self) -> None:
        self.assertTrue(is_proactive_message('{"proactive": true}'))
        self.assertFalse(is_proactive_message('{"proactive": false}'))

    def test_empty(self) -> None:
        self.assertFalse(is_proactive_message(None))
        self.assertFalse(is_proactive_message(""))


class TokenizeTest(unittest.TestCase):
    def test_chinese_chars(self) -> None:
        toks = tokenize("今天 天气")
        self.assertIn("今", toks)
        self.assertIn("天", toks)
        self.assertIn("气", toks)

    def test_english_words(self) -> None:
        toks = tokenize("hello world python")
        self.assertIn("hello", toks)
        self.assertIn("world", toks)


class ScoreFeedbackTest(unittest.TestCase):
    def test_explicit_quote_high_confidence(self) -> None:
        # 用户把推送原文当自己的回复（无 marker，完全照搬）
        s = score_feedback(
            "今天天气真好",
            "今天天气真好",
            candidates=["今天天气真好", "另一条候选"],
        )
        self.assertEqual(s.feedback_type, "explicit_quote")
        self.assertEqual(s.confidence, "high")
        self.assertEqual(s.matched_by, "exact")
        self.assertGreaterEqual(s.pa_score, EXACT_MATCH_THRESHOLD)

    def test_topic_follow(self) -> None:
        # 话题有关但措辞不同：推送谈天气，用户回应里也有"天气"
        s = score_feedback(
            "今天上海的天气真好",
            "是啊 天气真的不错",
        )
        self.assertEqual(s.feedback_type, "topic_follow")
        self.assertIn(s.matched_by, {"token", "topic"})

    def test_no_topic_follow(self) -> None:
        s = score_feedback(
            "Python 类型注解很好用",
            "今天我们来聊聊机器学习模型训练",
        )
        self.assertEqual(s.feedback_type, "no_topic_follow")
        self.assertEqual(s.confidence, "low")
        self.assertEqual(s.matched_by, "none")

    def test_neutral_when_no_marker_and_no_overlap(self) -> None:
        s = score_feedback(
            "数据库优化方法",
            "完全无关的内容 hello world",
        )
        self.assertEqual(s.feedback_type, "no_topic_follow")

    def test_lag_seconds_computed(self) -> None:
        from datetime import datetime, timedelta, timezone

        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=30)
        s = score_feedback(
            "推送",
            "用户回复",
            proactive_ts=t0.isoformat(),
            user_ts=t1.isoformat(),
        )
        self.assertEqual(s.lag_seconds, 30)

    def test_pua_score_is_inverse_of_pa(self) -> None:
        s = score_feedback("x", "x", candidates=["x"])
        self.assertGreater(s.pa_score, 0.5)
        self.assertLess(s.pua_score, 0.5)

    def test_candidate_count_tracked(self) -> None:
        s = score_feedback(
            "x", "y", candidates=["a", "b", "c", "d"],
        )
        self.assertEqual(s.candidate_count, 4)


class FeedbackDbTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.conn = open_db(Path(self._tmp.name) / "fb.db")

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def _event(self, *, session="s1", user_id="u1", proactive_id="p1", ft="explicit_quote", confidence="high"):
        return FeedbackEvent(
            session_key=session,
            user_message_id=user_id,
            assistant_message_id="a1",
            proactive_message_id=proactive_id,
            feedback_type=ft,
            confidence="high",
            pa_score=0.9,
            pua_score=0.1,
            lag_seconds=5,
            candidate_count=2,
            matched_by="exact",
            reason="test",
        )

    def test_insert_and_readback(self) -> None:
        insert_feedback(self.conn, self._event())
        rows = list_session_events(self.conn, "s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feedback_type"], "explicit_quote")
        self.assertEqual(rows[0]["session_key"], "s1")

    def test_same_user_message_replaces(self) -> None:
        insert_feedback(self.conn, self._event(ft="explicit_quote"))
        insert_feedback(self.conn, self._event(ft="topic_follow", confidence="medium"))
        rows = list_session_events(self.conn, "s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feedback_type"], "topic_follow")

    def test_events_for_proactive(self) -> None:
        insert_feedback(self.conn, self._event(user_id="u1", proactive_id="px"))
        insert_feedback(self.conn, self._event(user_id="u2", proactive_id="px"))
        rows = events_for_proactive(self.conn, "px")
        self.assertEqual(len(rows), 2)

    def test_event_count_by_type(self) -> None:
        insert_feedback(self.conn, self._event(user_id="u1", ft="explicit_quote"))
        insert_feedback(self.conn, self._event(user_id="u2", ft="topic_follow"))
        insert_feedback(self.conn, self._event(user_id="u3", ft="no_topic_follow"))
        counts = event_count_by_type(self.conn)
        self.assertEqual(counts["explicit_quote"], 1)
        self.assertEqual(counts["topic_follow"], 1)
        self.assertEqual(counts["no_topic_follow"], 1)

    def test_delete_session(self) -> None:
        insert_feedback(self.conn, self._event(session="s1", user_id="u1"))
        insert_feedback(self.conn, self._event(session="s2", user_id="u2"))
        n = delete_session_events(self.conn, "s1")
        self.assertEqual(n, 1)
        self.assertEqual(len(list_session_events(self.conn, "s1")), 0)
        self.assertEqual(len(list_session_events(self.conn, "s2")), 1)

    def test_avg_metric_by_type(self) -> None:
        insert_feedback(self.conn, self._event(user_id="u1", ft="explicit_quote"))
        insert_feedback(self.conn, self._event(user_id="u2", ft="explicit_quote"))
        insert_feedback(self.conn, self._event(user_id="u3", ft="topic_follow"))
        avgs = avg_metric_by_type(self.conn, "pa_score")
        self.assertAlmostEqual(avgs["explicit_quote"], 0.9, places=4)

    def test_avg_metric_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            avg_metric_by_type(self.conn, "invalid_metric")


class FeedbackReporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.reporter = FeedbackReporter.open(Path(self._tmp.name) / "fb.db")
        self.conn = self.reporter._conn

    def tearDown(self) -> None:
        self.reporter.close()
        self._tmp.cleanup()

    def _seed(self):
        insert_feedback(
            self.conn,
            FeedbackEvent(
                session_key="s1",
                user_message_id="u1",
                assistant_message_id="a1",
                proactive_message_id="p1",
                feedback_type="explicit_quote",
                confidence="high",
                pa_score=0.9,
                pua_score=0.1,
                lag_seconds=5,
                candidate_count=1,
                matched_by="exact",
                reason="test",
            ),
        )
        insert_feedback(
            self.conn,
            FeedbackEvent(
                session_key="s1",
                user_message_id="u2",
                assistant_message_id="a2",
                proactive_message_id="p1",
                feedback_type="topic_follow",
                confidence="high",
                pa_score=0.5,
                pua_score=0.5,
                lag_seconds=10,
                candidate_count=1,
                matched_by="token",
                reason="test",
            ),
        )
        insert_feedback(
            self.conn,
            FeedbackEvent(
                session_key="s2",
                user_message_id="u3",
                assistant_message_id="a3",
                proactive_message_id="p2",
                feedback_type="no_topic_follow",
                confidence="low",
                pa_score=0.1,
                pua_score=0.9,
                lag_seconds=20,
                candidate_count=1,
                matched_by="none",
                reason="test",
            ),
        )

    def test_summary_aggregates(self) -> None:
        self._seed()
        s = self.reporter.summary()
        self.assertEqual(s.total_events, 3)
        self.assertEqual(s.by_type["explicit_quote"], 1)
        self.assertEqual(s.by_type["topic_follow"], 1)
        self.assertEqual(s.by_type["no_topic_follow"], 1)
        self.assertEqual(s.sessions["s1"], 2)
        self.assertEqual(s.sessions["s2"], 1)

    def test_engagement_rate(self) -> None:
        self._seed()
        s = self.reporter.summary()
        self.assertAlmostEqual(s.engagement_rate(), 2 / 3, places=4)

    def test_deflection_rate(self) -> None:
        self._seed()
        s = self.reporter.summary()
        self.assertAlmostEqual(s.deflection_rate(), 1 / 3, places=4)

    def test_session_summary(self) -> None:
        self._seed()
        ss = self.reporter.session_summary("s1")
        self.assertEqual(ss["event_count"], 2)
        self.assertEqual(ss["by_type"]["explicit_quote"], 1)

    def test_empty_summary(self) -> None:
        s = self.reporter.summary()
        self.assertEqual(s.total_events, 0)
        self.assertEqual(s.engagement_rate(), 0.0)
        self.assertEqual(s.deflection_rate(), 0.0)


if __name__ == "__main__":
    unittest.main()