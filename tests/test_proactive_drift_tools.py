"""主动链路漂移工具集测试。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from initiative.proactive_drift_state import DriftStateStore
from initiative.proactive_drift_tools import (
    BoundedWebFetchTool,
    DriftRunContext,
    FinishDriftTool,
    MountServerTool,
    PushMessageTool,
    _normalize_media,
)


def _parse(text: str) -> dict:
    return json.loads(text)


class DriftRunContextTest(unittest.TestCase):
    def test_defaults(self) -> None:
        ctx = DriftRunContext()
        self.assertFalse(ctx.message_sent)
        self.assertFalse(ctx.finished)
        self.assertEqual(ctx.mounted_servers, set())
        self.assertIsNotNone(ctx.now_utc)

    def test_reset(self) -> None:
        ctx = DriftRunContext()
        ctx.message_sent = True
        ctx.finished = True
        ctx.mounted_servers.add("srv")
        ctx.reset()
        self.assertFalse(ctx.message_sent)
        self.assertFalse(ctx.finished)
        self.assertEqual(ctx.mounted_servers, set())


class NormalizeMediaTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(_normalize_media(), [])

    def test_image_only(self) -> None:
        self.assertEqual(_normalize_media(image="a.png"), ["a.png"])

    def test_media_string(self) -> None:
        self.assertEqual(_normalize_media(media="b.jpg"), ["b.jpg"])

    def test_media_list(self) -> None:
        self.assertEqual(
            _normalize_media(media=["c.png", "d.jpg"]),
            ["c.png", "d.jpg"],
        )

    def test_combined_and_filtered(self) -> None:
        out = _normalize_media(
            image=" a.png ", media=["", "b.jpg", "  "]
        )
        self.assertEqual(out, ["a.png", "b.jpg"])


class PushMessageToolTest(unittest.IsolatedAsyncioTestCase):
    def _make_sender(self, ok=True, calls=None):
        async def _send(text, media):
            if calls is not None:
                calls.append((text, media))
            return ok

        return _send

    async def test_success_sets_flag(self) -> None:
        calls: list = []
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender(calls=calls))
        result = _parse(await tool.execute({"message": "hi"}))
        self.assertTrue(result["ok"])
        self.assertTrue(ctx.message_sent)
        self.assertEqual(calls, [("hi", [])])

    async def test_media_sent(self) -> None:
        calls: list = []
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender(calls=calls))
        await tool.execute({"message": "看图", "media": ["a.png"]})
        self.assertEqual(calls, [("看图", ["a.png"])])

    async def test_image_and_media_combined(self) -> None:
        calls: list = []
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender(calls=calls))
        await tool.execute({"message": "", "image": "x.png", "media": ["y.png"]})
        self.assertEqual(calls, [("", ["x.png", "y.png"])])
        self.assertTrue(ctx.message_sent)

    async def test_not_configured(self) -> None:
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, None)
        result = _parse(await tool.execute({"message": "hi"}))
        self.assertIn("not configured", result["error"])

    async def test_second_call_rejected(self) -> None:
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender())
        await tool.execute({"message": "first"})
        result = _parse(await tool.execute({"message": "second"}))
        self.assertIn("already used", result["error"])

    async def test_empty_message_and_media_rejected(self) -> None:
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender())
        result = _parse(await tool.execute({"message": "  "}))
        self.assertIn("required", result["error"])
        self.assertFalse(ctx.message_sent)

    async def test_sender_returns_false(self) -> None:
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender(ok=False))
        result = _parse(await tool.execute({"message": "hi"}))
        self.assertIn("failed", result["error"])
        self.assertFalse(ctx.message_sent)

    async def test_sender_raises(self) -> None:
        async def _boom(text, media):
            raise RuntimeError("channel down")

        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, _boom)
        result = _parse(await tool.execute({"message": "hi"}))
        self.assertIn("failed", result["error"])
        self.assertFalse(ctx.message_sent)

    async def test_content_alias(self) -> None:
        calls: list = []
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, self._make_sender(calls=calls))
        await tool.execute({"content": "用 content 字段"})
        self.assertEqual(calls, [("用 content 字段", [])])

    def test_schema_shape(self) -> None:
        ctx = DriftRunContext()
        tool = PushMessageTool(ctx, None)
        schema = tool.to_schema()
        self.assertEqual(schema["function"]["name"], "message_push")
        self.assertIn("message", schema["function"]["parameters"]["properties"])


class FinishDriftToolTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = DriftStateStore(self.root)
        # 造一个合法 skill
        skill_dir = self.root / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo\ndescription: test skill\n---\nbody",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _tool(self, ctx):
        return FinishDriftTool(ctx, self.store)

    async def test_successful_silent_finish(self) -> None:
        ctx = DriftRunContext(
            now_utc=datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        )
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "完成分析",
                "next": "继续深入",
                "message_result": "silent",
            })
        )
        self.assertTrue(result["ok"])
        self.assertTrue(ctx.finished)
        drift = self.store.load_drift()
        self.assertEqual(len(drift["recent_runs"]), 1)
        self.assertEqual(drift["recent_runs"][0]["message_result"], "silent")

    async def test_successful_sent_finish(self) -> None:
        ctx = DriftRunContext()
        ctx.message_sent = True  # 模拟已成功 push
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "done",
                "next": "n",
                "message_result": "sent",
            })
        )
        self.assertTrue(result["ok"])

    async def test_unknown_skill_rejected(self) -> None:
        ctx = DriftRunContext()
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "nope",
                "one_line": "x",
                "next": "y",
                "message_result": "silent",
            })
        )
        self.assertIn("unknown skill", result["error"])
        self.assertFalse(ctx.finished)

    async def test_missing_one_line(self) -> None:
        ctx = DriftRunContext()
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "",
                "next": "y",
                "message_result": "silent",
            })
        )
        self.assertIn("one_line", result["error"])

    async def test_missing_next(self) -> None:
        ctx = DriftRunContext()
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "x",
                "next": "",
                "message_result": "silent",
            })
        )
        self.assertIn("next", result["error"])

    async def test_bad_message_result_value(self) -> None:
        ctx = DriftRunContext()
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "x",
                "next": "y",
                "message_result": "maybe",
            })
        )
        self.assertIn("sent, silent", result["error"])

    async def test_sent_without_push_rejected(self) -> None:
        ctx = DriftRunContext()
        self.assertFalse(ctx.message_sent)
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "x",
                "next": "y",
                "message_result": "sent",
            })
        )
        self.assertIn("requires successful message_push", result["error"])

    async def test_silent_with_push_rejected(self) -> None:
        ctx = DriftRunContext()
        ctx.message_sent = True
        result = _parse(
            await self._tool(ctx).execute({
                "skill_used": "demo",
                "one_line": "x",
                "next": "y",
                "message_result": "silent",
            })
        )
        self.assertIn("conflicts", result["error"])

    async def test_note_saved_when_provided(self) -> None:
        ctx = DriftRunContext()
        await self._tool(ctx).execute({
            "skill_used": "demo",
            "one_line": "x",
            "next": "y",
            "message_result": "silent",
            "note": "备注内容",
        })
        drift = self.store.load_drift()
        self.assertEqual(drift["note"], "备注内容")

    def test_schema_required_fields(self) -> None:
        schema = FinishDriftTool(DriftRunContext(), self.store).to_schema()
        required = schema["function"]["parameters"]["required"]
        self.assertIn("skill_used", required)
        self.assertIn("message_result", required)


class MountServerToolTest(unittest.IsolatedAsyncioTestCase):
    def _source(self, mapping: dict):
        return SimpleNamespace(
            list_server_tools=lambda server: mapping.get(server, []),
        )

    async def test_mount_new_server(self) -> None:
        mounted: set[str] = set()
        tool = MountServerTool(
            self._source({"fetch": ["mcp_fetch__get", "mcp_fetch__head"]}),
            mounted,
        )
        result = _parse(await tool.execute({"server": "fetch"}))
        self.assertTrue(result["ok"])
        self.assertEqual(result["new"], ["mcp_fetch__get", "mcp_fetch__head"])
        self.assertEqual(len(mounted), 2)

    async def test_empty_server_rejected(self) -> None:
        tool = MountServerTool(self._source({}), set())
        result = _parse(await tool.execute({"server": ""}))
        self.assertIn("required", result["error"])

    async def test_unknown_server(self) -> None:
        tool = MountServerTool(self._source({}), set())
        result = _parse(await tool.execute({"server": "ghost"}))
        self.assertIn("不存在", result["error"])

    async def test_already_mounted_idempotent(self) -> None:
        mounted = {"mcp_x__a"}
        tool = MountServerTool(
            self._source({"x": ["mcp_x__a"]}), mounted
        )
        result = _parse(await tool.execute({"server": "x"}))
        self.assertTrue(result["ok"])
        self.assertIn("已挂载", result["message"])
        self.assertNotIn("new", result)


class _FakeWebFetch:
    def __init__(self, payload: dict | str):
        self.name = "web_fetch"
        self.description = "抓取网页"
        self.parameters = {"type": "object", "properties": {}}
        self._payload = payload

    async def execute(self, arguments):
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload, ensure_ascii=False)


class BoundedWebFetchToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_short_text_untouched(self) -> None:
        inner = _FakeWebFetch({"text": "short"})
        tool = BoundedWebFetchTool(inner, max_chars=100)
        out = await tool.execute({})
        self.assertEqual(_parse(out)["text"], "short")
        self.assertNotIn("truncated", _parse(out))

    async def test_long_text_truncated(self) -> None:
        inner = _FakeWebFetch({"text": "x" * 500})
        tool = BoundedWebFetchTool(inner, max_chars=100)
        out = _parse(await tool.execute({}))
        self.assertEqual(len(out["text"]), 100)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["length"], 100)
        self.assertIn("截断", out["note"])

    async def test_non_json_passthrough(self) -> None:
        inner = _FakeWebFetch("plain text response")
        tool = BoundedWebFetchTool(inner, max_chars=5)
        out = await tool.execute({})
        self.assertEqual(out, "plain text response")

    async def test_json_without_text_passthrough(self) -> None:
        inner = _FakeWebFetch({"other": "x" * 500})
        tool = BoundedWebFetchTool(inner, max_chars=5)
        out = _parse(await tool.execute({}))
        self.assertNotIn("truncated", out)

    async def test_delegates_name_and_schema(self) -> None:
        inner = _FakeWebFetch({"text": ""})
        tool = BoundedWebFetchTool(inner, max_chars=10)
        self.assertEqual(tool.name, "web_fetch")
        schema = tool.to_schema()
        self.assertEqual(schema["function"]["name"], "web_fetch")

    async def test_max_chars_floor_one(self) -> None:
        inner = _FakeWebFetch({"text": "ab"})
        tool = BoundedWebFetchTool(inner, max_chars=0)
        out = _parse(await tool.execute({}))
        self.assertEqual(len(out["text"]), 1)


if __name__ == "__main__":
    unittest.main()
