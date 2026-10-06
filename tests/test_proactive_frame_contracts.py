"""主动链路 Frame + Contracts 数据契约测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from initiative.proactive_frame import (
    ProactiveFrame,
    ProactiveTickInput,
    ProactiveTickResult,
    new_frame,
    new_proactive_frame,
)
from initiative.proactive_contracts import (
    MAX_METRICS_KEYS,
    MAX_METRICS_VALUE_STR_LEN,
    format_local_time,
    looks_like_time_key,
    normalize_metrics,
    resolve_tz,
    trim_text,
)


class ProactiveTickInputTest(unittest.TestCase):
    def test_input_is_frozen(self) -> None:
        inp = ProactiveTickInput(session_key="s1", started_at=datetime.now(timezone.utc))
        with self.assertRaises(Exception):
            inp.session_key = "s2"  # type: ignore[misc]

    def test_fields_roundtrip(self) -> None:
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        inp = ProactiveTickInput(session_key="s1", started_at=ts)
        self.assertEqual(inp.session_key, "s1")
        self.assertEqual(inp.started_at, ts)


class ProactiveTickResultTest(unittest.TestCase):
    def test_default_score_none(self) -> None:
        r = ProactiveTickResult()
        self.assertIsNone(r.base_score)

    def test_set_score(self) -> None:
        r = ProactiveTickResult(base_score=0.73)
        self.assertAlmostEqual(r.base_score, 0.73)


class NewFrameTest(unittest.TestCase):
    def test_defaults(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        f = new_frame("s1", now=ts)
        self.assertEqual(f.input.session_key, "s1")
        self.assertEqual(f.input.started_at, ts)
        self.assertEqual(f.slots, {})
        self.assertIsNone(f.output)

    def test_with_slots(self) -> None:
        f = new_frame("s1", slots={"base_score": 0.5, "tone": "warm"})
        self.assertEqual(f.slots["base_score"], 0.5)
        self.assertEqual(f.slots["tone"], "warm")

    def test_slots_isolated_between_frames(self) -> None:
        slots = {"k": 1}
        f1 = new_frame("s1", slots=slots)
        slots["k"] = 999  # mutate caller dict
        f2 = new_frame("s2", slots=slots)
        # f1 的 slots 不应受 caller dict 后续变化影响
        self.assertEqual(f1.slots["k"], 1)
        self.assertEqual(f2.slots["k"], 999)

    def test_now_default_is_recent_utc(self) -> None:
        before = datetime.now(timezone.utc)
        f = new_frame("s1")
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(f.input.started_at, before - timedelta(seconds=1))
        self.assertLessEqual(f.input.started_at, after + timedelta(seconds=1))

    def test_legacy_alias(self) -> None:
        f = new_proactive_frame("s1")
        self.assertIsInstance(f, ProactiveFrame)


class FrameMutabilityTest(unittest.TestCase):
    def test_slots_mutable_across_modules(self) -> None:
        """模块间共享 slots：模块 A 写入后模块 B 可读。"""
        f = new_frame("s1")
        f.slots["base_score"] = 0.42
        f.slots["prompt_section"] = "问候"
        self.assertEqual(f.slots["base_score"], 0.42)
        self.assertEqual(f.slots["prompt_section"], "问候")

    def test_output_writable_later(self) -> None:
        f = new_frame("s1")
        self.assertIsNone(f.output)
        f.output = ProactiveTickResult(base_score=0.88)
        self.assertAlmostEqual(f.output.base_score, 0.88)


class TrimTextTest(unittest.TestCase):
    def test_under_limit_unchanged(self) -> None:
        self.assertEqual(trim_text("hello", 10), "hello")

    def test_at_limit_unchanged(self) -> None:
        self.assertEqual(trim_text("x" * 10, 10), "x" * 10)

    def test_over_limit_truncated_with_marker(self) -> None:
        out = trim_text("x" * 100, 10)
        self.assertEqual(len(out), 10 + 3)  # "..." adds 3 chars
        self.assertTrue(out.endswith("..."))


class NormalizeMetricsTest(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(normalize_metrics(None))

    def test_empty_dict_returns_none(self) -> None:
        self.assertIsNone(normalize_metrics({}))

    def test_non_dict_returns_none(self) -> None:
        self.assertIsNone(normalize_metrics(["list"]))  # type: ignore[arg-type]

    def test_keeps_basic_types(self) -> None:
        result = normalize_metrics({"count": 5, "ratio": 0.7, "flag": True})
        self.assertEqual(result, {"count": 5, "ratio": 0.7, "flag": True})

    def test_string_truncated(self) -> None:
        result = normalize_metrics({"text": "x" * (MAX_METRICS_VALUE_STR_LEN + 50)})
        self.assertIsNotNone(result)
        self.assertEqual(len(result["text"]), MAX_METRICS_VALUE_STR_LEN + 3)
        self.assertTrue(result["text"].endswith("..."))

    def test_complex_serialized_to_json(self) -> None:
        result = normalize_metrics({"data": {"nested": [1, 2, 3]}})
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], '{"nested": [1, 2, 3]}')

    def test_exceeds_key_count_records_truncated(self) -> None:
        items = {f"k{i}": i for i in range(MAX_METRICS_KEYS + 5)}
        result = normalize_metrics(items)
        self.assertIsNotNone(result)
        self.assertIn("_truncated_keys", result)
        self.assertEqual(result["_truncated_keys"], 5)

    def test_blank_keys_skipped(self) -> None:
        result = normalize_metrics({"   ": 1, "good": 2, "": 3})
        self.assertIsNotNone(result)
        self.assertNotIn("", result)
        self.assertNotIn("   ", result)
        self.assertIn("good", result)


class LooksLikeTimeKeyTest(unittest.TestCase):
    def test_whitelisted_keys(self) -> None:
        for k in ("last_seen", "updated_at", "published_at", "timestamp", "ts"):
            self.assertTrue(looks_like_time_key(k))

    def test_suffix_based_keys(self) -> None:
        for k in ("created_at", "edited_time", "expired_ts", "foo_at"):
            self.assertTrue(looks_like_time_key(k))

    def test_random_keys(self) -> None:
        for k in ("value", "count", "ratio", "name"):
            self.assertFalse(looks_like_time_key(k))


class ResolveTzTest(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(resolve_tz(None))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(resolve_tz(""))

    def test_whitespace_string_returns_none(self) -> None:
        self.assertIsNone(resolve_tz("   "))

    def test_invalid_zone_string_returns_none(self) -> None:
        self.assertIsNone(resolve_tz("Invalid/Zone"))

    def test_valid_utc_zone(self) -> None:
        tz = resolve_tz("UTC")
        self.assertIsNotNone(tz)

    def test_tzinfo_passthrough(self) -> None:
        original = timezone.utc
        self.assertIs(resolve_tz(original), original)


class FormatLocalTimeTest(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(format_local_time(""))

    def test_naive_datetime_returns_none(self) -> None:
        self.assertIsNone(format_local_time("2026-01-01T12:00:00"))

    def test_aware_datetime(self) -> None:
        out = format_local_time("2026-01-01T12:00:00+00:00", "UTC")
        self.assertEqual(out, "2026-01-01 12:00:00 +0000")

    def test_invalid_string_returns_none(self) -> None:
        self.assertIsNone(format_local_time("not a datetime"))

    def test_with_zone_conversion(self) -> None:
        out = format_local_time("2026-01-01T00:00:00+00:00", "UTC")
        self.assertIn("+0000", out)
        self.assertIn("2026-01-01", out)


if __name__ == "__main__":
    unittest.main()
