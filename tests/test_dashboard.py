"""Dashboard 调试入口测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateways.dashboard import (
    _format_ts,
    _read_memory_facts,
    create_dashboard_app,
)


class FormatTimestampTest(unittest.TestCase):
    """ISO 时间戳格式化。"""

    def test_none_returns_never(self) -> None:
        self.assertEqual(_format_ts(None), "从未")

    def test_valid_iso(self) -> None:
        result = _format_ts("2026-09-07T12:34:56+00:00")
        self.assertIn("2026-09-07", result)
        self.assertIn("12:34:56", result)

    def test_invalid_returns_raw(self) -> None:
        self.assertEqual(_format_ts("not-a-date"), "not-a-date")


class ReadMemoryFactsTest(unittest.TestCase):
    """MEMORY.md 文件解析。"""

    def test_missing_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(_read_memory_facts(Path(temp_dir) / "x.md"), [])

    def test_extracts_bullet_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            md = Path(temp_dir) / "MEMORY.md"
            md.write_text(
                "# 长期记忆\n\n- 用户喜欢喝咖啡\n- 项目用 Python 3.10\n",
                encoding="utf-8",
            )
            facts = _read_memory_facts(md)
            self.assertEqual(facts, ["用户喜欢喝咖啡", "项目用 Python 3.10"])

    def test_ignores_non_bullet_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            md = Path(temp_dir) / "MEMORY.md"
            md.write_text(
                "# 标题\n\n正文段落\n- 事实 1\n## 子标题\n- 事实 2\n",
                encoding="utf-8",
            )
            facts = _read_memory_facts(md)
            self.assertEqual(facts, ["事实 1", "事实 2"])


class _FakeMemory:
    def __init__(self, pending: list[str] | None = None):
        self._pending = pending or []
        self.promote_pending_called = 0

    def unpromoted_pending(self) -> list[str]:
        return self._pending

    def promote_pending(self) -> list[str]:
        self.promote_pending_called += 1
        promoted = list(self._pending)
        self._pending = []
        return promoted


class _FakeSession:
    def __init__(self, messages: list[dict] | None = None):
        self.messages = messages or []


class _FakeStats:
    def __init__(self, summary: dict | None = None):
        self._summary = summary or {
            "total_turns": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
        }

    def summary(self) -> dict:
        return self._summary


class _FakeSessionStore:
    def list_sessions(self) -> list[dict]:
        return [
            {"session_id": "abc123def456", "message_count": 4},
            {"session_id": "xyz789ghi012", "message_count": 2},
        ]


class _FakeTools:
    def get_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "检索记忆",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_fake__echo",
                    "description": "[MCP:fake] 回显",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]


class _FakeMcpClient:
    def __init__(self, connected: bool, tools: list) -> None:
        self.is_connected = connected
        self.tools = tools


class _FakeMcpRegistry:
    def __init__(self, clients: dict) -> None:
        self._clients = clients


class _FakeAgent:
    def __init__(self):
        self._session_id = "test-session-1"
        self._session = _FakeSession(
            messages=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我能帮你什么？"},
                {"role": "user", "content": "帮我看下文件"},
            ]
        )
        self._memory = _FakeMemory(pending=["事实 1", "事实 2"])
        self._stats = _FakeStats(
            summary={
                "total_turns": 5,
                "total_tokens": 1234,
                "avg_latency_ms": 456.7,
        })
        self._session_store = _FakeSessionStore()
        self._tools = _FakeTools()
        self._mcp_registry = _FakeMcpRegistry({
            "fake": _FakeMcpClient(
                connected=True,
                tools=[
                    SimpleNamespace(name="echo", description="回显"),
                ],
            ),
            "down": _FakeMcpClient(connected=False, tools=[]),
        })


class DashboardEndpointsTest(unittest.IsolatedAsyncioTestCase):
    """Dashboard 路由端到端测试。"""

    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self._temp_dir = Path(self._temp_dir_ctx.__enter__())
        self.agent = _FakeAgent()
        self.presence = MagicMock()
        self.presence.get_last_user_at = MagicMock(return_value=None)
        self.presence.get_last_proactive_at = MagicMock(return_value=None)
        self.app = create_dashboard_app(self.agent, self.presence, self._temp_dir)

    def tearDown(self) -> None:
        self._temp_dir_ctx.__exit__(None, None, None)

    def _make_app(self) -> tuple:
        """向后兼容的工厂，返回 (app, agent, workspace)。"""
        return self.app, self.agent, self._temp_dir

    async def test_index_returns_html(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/")
        self.assertEqual(resp.status_code, 200)
        text = resp.text
        self.assertIn("ProactiveMind Dashboard", text)
        self.assertIn("/api/status", text)

    async def test_api_status_returns_session_info(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], "test-session-1")
        self.assertEqual(data["user_message_count"], 2)
        self.assertEqual(data["last_user_message"], "帮我看下文件")

    async def test_api_presence_returns_timestamps(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/presence")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("last_user_at", data)
        self.assertIn("last_proactive_at", data)
        self.assertIn("now", data)

    async def test_api_stats_returns_summary(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_turns"], 5)
        self.assertEqual(data["total_tokens"], 1234)

    async def test_api_stats_empty(self) -> None:
        app, agent, _ = self._make_app()
        agent._stats = _FakeStats()  # 0 turns
        client = _make_async_client(app)
        resp = await client.get("/api/stats")
        data = resp.json()
        self.assertTrue(data["empty"])

    async def test_api_sessions_returns_list(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/sessions")
        data = resp.json()
        self.assertEqual(len(data["sessions"]), 2)

    async def test_api_memory_returns_pending_and_memory(self) -> None:
        app, agent, workspace = self._make_app()
        # 写入 MEMORY.md
        (workspace / "MEMORY.md").write_text(
            "- 长期事实\n- 偏好\n",
            encoding="utf-8",
        )

        client = _make_async_client(app)
        resp = await client.get("/api/memory")
        data = resp.json()
        self.assertEqual(data["pending"], ["事实 1", "事实 2"])
        self.assertEqual(data["memory"], ["长期事实", "偏好"])

    async def test_api_memory_handles_missing_file(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/memory")
        data = resp.json()
        self.assertEqual(data["pending"], ["事实 1", "事实 2"])
        self.assertEqual(data["memory"], [])

    async def test_api_promote_returns_count(self) -> None:
        app, agent, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.post("/api/promote")
        data = resp.json()
        self.assertEqual(data["promoted"], 2)
        self.assertEqual(len(data["facts"]), 2)
        # agent._memory.promote_pending 应被调用
        self.assertEqual(agent._memory.promote_pending_called, 1)

    async def test_api_tools_lists_builtin_and_mcp(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/tools")
        data = resp.json()
        self.assertEqual(data["total"], 2)
        by_name = {t["name"]: t for t in data["tools"]}
        self.assertEqual(by_name["recall"]["source"], "builtin")
        self.assertEqual(by_name["mcp_fake__echo"]["source"], "mcp")

    async def test_api_mcp_reports_server_status(self) -> None:
        app, _, _ = self._make_app()
        client = _make_async_client(app)
        resp = await client.get("/api/mcp")
        data = resp.json()
        self.assertTrue(data["enabled"])
        by_name = {s["name"]: s for s in data["servers"]}
        self.assertTrue(by_name["fake"]["connected"])
        self.assertEqual(by_name["fake"]["tool_count"], 1)
        self.assertEqual(by_name["fake"]["tools"][0]["name"], "echo")
        self.assertFalse(by_name["down"]["connected"])
        self.assertEqual(by_name["down"]["tools"], [])

    async def test_api_mcp_disabled_when_no_registry(self) -> None:
        app, agent, _ = self._make_app()
        agent._mcp_registry = None
        client = _make_async_client(app)
        resp = await client.get("/api/mcp")
        data = resp.json()
        self.assertFalse(data["enabled"])
        self.assertEqual(data["servers"], [])


def _make_async_client(app):
    """构造 httpx AsyncClient，绑到 ASGI 应用。"""
    import httpx

    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


if __name__ == "__main__":
    unittest.main()