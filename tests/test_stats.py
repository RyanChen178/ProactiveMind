"""Turn 统计测试。"""

from __future__ import annotations

import unittest

from mind.stats import TurnStats, TurnRecord


class TurnStatsTest(unittest.TestCase):
    def test_records_turn_with_usage_and_latency(self) -> None:
        stats = TurnStats()
        record = stats.record(
            session_id="s1",
            user_input="你好",
            assistant_reply="你好呀",
            tool_calls=["get_time"],
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=1234.5,
        )
        self.assertEqual(record.turn_id, 1)
        self.assertEqual(record.prompt_tokens, 100)
        self.assertEqual(record.completion_tokens, 50)
        self.assertEqual(record.total_tokens, 150)
        self.assertEqual(record.tool_calls, ["get_time"])
        self.assertEqual(record.latency_ms, 1234.5)
        self.assertEqual(len(stats.records), 1)

    def test_summary_aggregates_multiple_turns(self) -> None:
        stats = TurnStats()
        stats.record(
            session_id="s1", user_input="a", assistant_reply="b",
            tool_calls=["t1"],
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=1000,
        )
        stats.record(
            session_id="s1", user_input="c", assistant_reply="d",
            tool_calls=["t1", "t2"],
            usage={"prompt_tokens": 200, "completion_tokens": 100},
            latency_ms=3000,
        )
        s = stats.summary()
        self.assertEqual(s["total_turns"], 2)
        self.assertEqual(s["total_tokens"], 450)
        self.assertEqual(s["total_tool_calls"], 3)
        self.assertEqual(s["avg_latency_ms"], 2000.0)

    def test_summary_empty_when_no_records(self) -> None:
        stats = TurnStats()
        s = stats.summary()
        self.assertEqual(s["total_turns"], 0)
        self.assertEqual(s["total_tokens"], 0)

    def test_recent_returns_last_n(self) -> None:
        stats = TurnStats()
        for i in range(5):
            stats.record(
                session_id="s1",
                user_input=f"q{i}",
                assistant_reply=f"a{i}",
            )
        recent = stats.recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].user_input, "q2")
        self.assertEqual(recent[2].user_input, "q4")

    def test_recent_empty_returns_all(self) -> None:
        stats = TurnStats()
        for i in range(3):
            stats.record(
                session_id="s1",
                user_input=f"q{i}",
                assistant_reply=f"a{i}",
            )
        self.assertEqual(len(stats.recent(0)), 0)

    def test_truncates_long_input(self) -> None:
        stats = TurnStats()
        long_text = "x" * 500
        record = stats.record(
            session_id="s1",
            user_input=long_text,
            assistant_reply=long_text,
        )
        self.assertEqual(len(record.user_input), 200)
        self.assertEqual(len(record.assistant_reply), 200)

    def test_evicts_old_records_when_exceeding_max(self) -> None:
        stats = TurnStats(max_records=3)
        for i in range(5):
            stats.record(
                session_id="s1",
                user_input=f"q{i}",
                assistant_reply=f"a{i}",
            )
        self.assertEqual(len(stats.records), 3)
        self.assertEqual(stats.records[0].user_input, "q2")
        self.assertEqual(stats.records[2].user_input, "q4")

    def test_records_property_returns_copy(self) -> None:
        stats = TurnStats()
        stats.record(session_id="s1", user_input="q", assistant_reply="a")
        records = stats.records
        records.clear()
        self.assertEqual(len(stats.records), 1)

    def test_handles_missing_usage_keys(self) -> None:
        stats = TurnStats()
        record = stats.record(
            session_id="s1",
            user_input="q",
            assistant_reply="a",
            usage=None,
        )
        self.assertEqual(record.prompt_tokens, 0)
        self.assertEqual(record.completion_tokens, 0)


if __name__ == "__main__":
    unittest.main()
