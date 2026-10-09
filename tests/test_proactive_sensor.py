"""主动链路 Sensor 感知层测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from initiative.proactive_sensor import (
    CONTENT_PREVIEW_CHARS,
    ProactiveSensor,
    RecentProactiveMessage,
)


def _make_session(messages):
    """构造最简 session-like 对象。"""
    return SimpleNamespace(messages=messages)


def _make_sessions(session):
    """构造最简 sessions-like API。"""
    return SimpleNamespace(get_or_create=lambda key: session)


def _make_cfg(channel="web", chat_id="u1", recent=20):
    return SimpleNamespace(
        default_channel=channel,
        default_chat_id=chat_id,
        recent_chat_messages=recent,
    )


def _ts(text):
    return text  # 保留为字符串


class TargetSessionKeyTest(unittest.TestCase):
    def test_normal(self) -> None:
        s = ProactiveSensor(_make_cfg("web", "u1"), _make_sessions(_make_session([])))
        self.assertEqual(s.target_session_key(), "web:u1")

    def test_missing_channel(self) -> None:
        s = ProactiveSensor(_make_cfg("", "u1"), _make_sessions(_make_session([])))
        self.assertEqual(s.target_session_key(), "")

    def test_missing_chat_id(self) -> None:
        s = ProactiveSensor(_make_cfg("web", ""), _make_sessions(_make_session([])))
        self.assertEqual(s.target_session_key(), "")

    def test_stripped(self) -> None:
        s = ProactiveSensor(_make_cfg("  web  ", "  u1  "), _make_sessions(_make_session([])))
        self.assertEqual(s.target_session_key(), "web:u1")


class LastUserAtTest(unittest.TestCase):
    def test_no_presence_returns_none(self) -> None:
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session([])), presence=None)
        self.assertIsNone(s.last_user_at())

    def test_no_target_key_returns_none(self) -> None:
        presence = SimpleNamespace(get_last_user_at=lambda k: datetime(2026, 1, 1, tzinfo=timezone.utc))
        s = ProactiveSensor(_make_cfg("", ""), _make_sessions(_make_session([])), presence=presence)
        self.assertIsNone(s.last_user_at())

    def test_delegates_to_presence(self) -> None:
        ts = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        presence = SimpleNamespace(get_last_user_at=lambda k: ts if k == "web:u1" else None)
        s = ProactiveSensor(_make_cfg("web", "u1"), _make_sessions(_make_session([])), presence=presence)
        self.assertEqual(s.last_user_at(), ts)


class CollectRecentTest(unittest.TestCase):
    def test_empty_session(self) -> None:
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session([])))
        self.assertEqual(s.collect_recent(), [])

    def test_no_target_key(self) -> None:
        s = ProactiveSensor(_make_cfg("", ""), _make_sessions(_make_session([])))
        self.assertEqual(s.collect_recent(), [])

    def test_session_lookup_failure(self) -> None:
        def _raise(key):
            raise RuntimeError("db down")

        s = ProactiveSensor(_make_cfg(), SimpleNamespace(get_or_create=_raise))
        self.assertEqual(s.collect_recent(), [])

    def test_filters_non_user_assistant(self) -> None:
        msgs = [
            {"role": "system", "content": "x"},
            {"role": "tool", "content": "x"},
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        self.assertEqual(s.collect_recent(), [])

    def test_skips_empty_content(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "  "},  # 全空白
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "hello")

    def test_respects_recent_limit(self) -> None:
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(50)]
        s = ProactiveSensor(_make_cfg("w", "u", recent=5), _make_sessions(_make_session(msgs)))
        results = s.collect_recent()
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["content"], "m45")

    def test_content_truncated(self) -> None:
        long_text = "x" * 500
        msgs = [{"role": "user", "content": long_text}]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent()
        self.assertEqual(len(results[0]["content"]), CONTENT_PREVIEW_CHARS)


class CollectRecentProactiveTest(unittest.TestCase):
    def test_only_proactive_assistant(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "non-proactive reply", "proactive": False},
            {"role": "assistant", "content": "proactive push", "proactive": True, "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent_proactive(n=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "proactive push")
        self.assertEqual(results[0].state_summary_tag, "none")

    def test_returns_in_time_order(self) -> None:
        msgs = [
            {"role": "assistant", "content": "early", "proactive": True, "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "late", "proactive": True, "timestamp": "2026-01-02T00:00:00+00:00"},
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent_proactive(n=5)
        self.assertEqual([r.content for r in results], ["early", "late"])

    def test_n_limits_count(self) -> None:
        msgs = [
            {"role": "assistant", "content": f"m{i}", "proactive": True, "timestamp": f"2026-01-{1 + i:02d}T00:00:00+00:00"}
            for i in range(10)
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent_proactive(n=3)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.content for r in results], ["m7", "m8", "m9"])

    def test_skips_empty_proactive_content(self) -> None:
        msgs = [
            {"role": "assistant", "content": "", "proactive": True},
            {"role": "assistant", "content": "real", "proactive": True, "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
        s = ProactiveSensor(_make_cfg(), _make_sessions(_make_session(msgs)))
        results = s.collect_recent_proactive()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "real")


class ParseTimestampTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = ProactiveSensor(_make_cfg(), _make_sessions(_make_session([])))

    def test_valid_iso(self) -> None:
        ts = self.sensor._parse_timestamp("2026-01-01T12:00:00+00:00")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.tzinfo, timezone.utc)

    def test_naive_gets_local_tz(self) -> None:
        ts = self.sensor._parse_timestamp("2026-01-01T12:00:00")
        self.assertIsNotNone(ts)
        self.assertIsNotNone(ts.tzinfo)

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(self.sensor._parse_timestamp("not a date"))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(self.sensor._parse_timestamp(""))


if __name__ == "__main__":
    unittest.main()
