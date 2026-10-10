"""JSON 抽取与主动推送去重测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from initiative.json_extract import extract_json_object, extract_json_text
from initiative.proactive_deduper import (
    MessageDeduper,
    _field,
    _recent_meta,
    format_recent_entries,
)


class ExtractJsonTextTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(extract_json_text('{"a": 1}'), '{"a": 1}')

    def test_markdown_fenced(self) -> None:
        text = "```json\n{\"a\": 1}\n```"
        self.assertEqual(extract_json_text(text), '{"a": 1}')

    def test_fenced_no_lang(self) -> None:
        text = "```\n{\"a\": 1}\n```"
        self.assertEqual(extract_json_text(text), '{"a": 1}')

    def test_with_surrounding_text(self) -> None:
        text = '好的，这是 JSON：{"a": 1}，请采纳'
        self.assertEqual(extract_json_text(text), '{"a": 1}')

    def test_nested_braces(self) -> None:
        text = '{"outer": {"inner": 1}}'
        self.assertEqual(extract_json_text(text), '{"outer": {"inner": 1}}')

    def test_no_json(self) -> None:
        self.assertEqual(extract_json_text("hello world"), "hello world")

    def test_empty(self) -> None:
        self.assertEqual(extract_json_text(""), "")


class ExtractJsonObjectTest(unittest.TestCase):
    def test_valid_dict(self) -> None:
        self.assertEqual(
            extract_json_object('{"a": 1, "b": "x"}'),
            {"a": 1, "b": "x"},
        )

    def test_strips_markdown(self) -> None:
        text = "```json\n{\"a\": 2}\n```"
        self.assertEqual(extract_json_object(text), {"a": 2})

    def test_rejects_list(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("[1, 2, 3]")

    def test_rejects_scalar(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("42")

    def test_rejects_invalid_json(self) -> None:
        with self.assertRaises(Exception):
            extract_json_object("{not valid}")


class FieldHelperTest(unittest.TestCase):
    def test_dict_access(self) -> None:
        self.assertEqual(_field({"name": "x"}, "name"), "x")
        self.assertEqual(_field({"name": None}, "name", "d"), "d")

    def test_object_access(self) -> None:
        obj = SimpleNamespace(name="y")
        self.assertEqual(_field(obj, "name"), "y")

    def test_missing_returns_default(self) -> None:
        self.assertEqual(_field({}, "missing", "fallback"), "fallback")


class RecentMetaTest(unittest.TestCase):
    def test_dict_with_timestamp(self) -> None:
        meta = _recent_meta(
            {"timestamp": "2026-01-01T12:00:00", "state_summary_tag": "warm"}
        )
        self.assertIn("time=2026-01-01T12:00:00", meta)
        self.assertIn("state_tag=warm", meta)

    def test_tag_none_excluded(self) -> None:
        meta = _recent_meta({"state_summary_tag": "none"})
        self.assertEqual(meta, [])

    def test_tag_default(self) -> None:
        meta = _recent_meta({})
        self.assertEqual(meta, [])


class FormatRecentEntriesTest(unittest.TestCase):
    def test_indexed_listing(self) -> None:
        out = format_recent_entries([
            {"content": "first"},
            {"content": "second"},
        ])
        self.assertIn("[1] first", out)
        self.assertIn("[2] second", out)
        self.assertIn("\n---\n", out)

    def test_skips_empty_content(self) -> None:
        out = format_recent_entries([
            {"content": "real"},
            {"content": ""},
            {"content": "another"},
        ])
        self.assertIn("[1] real", out)
        # 空 content 保留原序号（[2]），下一个非空继续递增为 [3]
        self.assertIn("[3] another", out)
        self.assertNotIn("[2] ", out)

    def test_with_metadata(self) -> None:
        out = format_recent_entries([
            {"content": "warm", "state_summary_tag": "warm"},
        ])
        self.assertIn("[1]", out)
        self.assertIn("warm", out)
        self.assertIn("state_tag=warm", out)

    def test_empty(self) -> None:
        self.assertEqual(format_recent_entries([]), "")


class MessageDeduperTest(unittest.IsolatedAsyncioTestCase):
    def _fake_response(self, content: str):
        return SimpleNamespace(content=content)

    def _make_provider(self, response_content: str):
        provider = SimpleNamespace()

        async def _chat(messages, tools=None, model=None, max_tokens=None):
            return self._fake_response(response_content)

        provider.chat = _chat
        return provider

    async def test_empty_recent_always_passes(self) -> None:
        provider = self._make_provider("{}")
        deduper = MessageDeduper(provider=provider, model="m", max_tokens=100)
        is_dup, reason = await deduper.is_duplicate("hello", recent_proactive=[])
        self.assertFalse(is_dup)
        self.assertIn("放行", reason)

    async def test_llm_says_duplicate(self) -> None:
        provider = self._make_provider(
            '{"is_duplicate": true, "reason": "语义雷同"}'
        )
        deduper = MessageDeduper(provider=provider, model="m", max_tokens=100)
        is_dup, reason = await deduper.is_duplicate(
            "hi", recent_proactive=[{"content": "hi"}]
        )
        self.assertTrue(is_dup)
        self.assertEqual(reason, "语义雷同")

    async def test_llm_says_not_duplicate(self) -> None:
        provider = self._make_provider(
            '{"is_duplicate": false, "reason": "无关话题"}'
        )
        deduper = MessageDeduper(provider=provider, model="m", max_tokens=100)
        is_dup, reason = await deduper.is_duplicate(
            "new thing",
            recent_proactive=[{"content": "old thing"}],
        )
        self.assertFalse(is_dup)
        self.assertEqual(reason, "无关话题")

    async def test_provider_error_passes_through(self) -> None:
        provider = SimpleNamespace()

        async def _chat(messages, tools=None, model=None, max_tokens=None):
            raise RuntimeError("provider offline")

        provider.chat = _chat
        deduper = MessageDeduper(provider=provider, model="m")
        is_dup, reason = await deduper.is_duplicate(
            "x", recent_proactive=[{"content": "y"}]
        )
        self.assertFalse(is_dup)
        self.assertIn("provider offline", reason)

    async def test_malformed_response_passes_through(self) -> None:
        provider = self._make_provider("not json at all")
        deduper = MessageDeduper(provider=provider, model="m")
        is_dup, reason = await deduper.is_duplicate(
            "x", recent_proactive=[{"content": "y"}]
        )
        self.assertFalse(is_dup)

    async def test_list_response_rejected_passes_through(self) -> None:
        provider = self._make_provider("[1, 2, 3]")
        deduper = MessageDeduper(provider=provider, model="m")
        is_dup, reason = await deduper.is_duplicate(
            "x", recent_proactive=[{"content": "y"}]
        )
        self.assertFalse(is_dup)

    async def test_prompt_contains_recent(self) -> None:
        captured: list[list[dict]] = []

        async def _chat(messages, tools=None, model=None, max_tokens=None):
            captured.append(messages)
            return self._fake_response('{"is_duplicate": false}')

        provider = SimpleNamespace(chat=_chat)
        deduper = MessageDeduper(provider=provider, model="m", max_tokens=200)
        await deduper.is_duplicate(
            "new msg",
            recent_proactive=[
                {"content": "old msg 1"},
                {"content": "old msg 2"},
            ],
        )
        user_msg = captured[0][1]["content"]
        self.assertIn("old msg 1", user_msg)
        self.assertIn("old msg 2", user_msg)
        self.assertIn("new msg", user_msg)

    async def test_max_tokens_capped(self) -> None:
        captured_max: list[int] = []

        async def _chat(messages, tools=None, model=None, max_tokens=None):
            captured_max.append(max_tokens)
            return self._fake_response('{"is_duplicate": false}')

        provider = SimpleNamespace(chat=_chat)
        deduper = MessageDeduper(provider=provider, model="m", max_tokens=999)
        await deduper.is_duplicate("x", recent_proactive=[{"content": "y"}])
        self.assertEqual(captured_max[0], 128)


if __name__ == "__main__":
    unittest.main()
