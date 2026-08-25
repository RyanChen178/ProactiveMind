"""会话管理 REST API 测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from starlette.testclient import TestClient

from channels.web_chat import create_app


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent._session_id = "test-session-1"

    async def _run_stream(user_input: str, max_steps: int = 10):
        yield "ok"

    agent.run_stream = _run_stream
    agent.stats.summary.return_value = {"total_turns": 0}
    agent.stats.recent.return_value = []
    agent.list_sessions.return_value = [
        {"id": "s1", "created_at": "2026-01-01T00:00:00Z", "message_count": 5, "is_active": True},
        {"id": "s2", "created_at": "2026-01-02T00:00:00Z", "message_count": 3, "is_active": False},
    ]
    agent.get_session_history.return_value = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]
    agent.switch_session.return_value = True
    agent.reset_session.return_value = None
    agent.export_session_markdown.return_value = "# 会话导出\n\n你好"
    return agent


class SessionAPITest(unittest.TestCase):
    def test_list_sessions(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.get("/sessions")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["sessions"]), 2)
        self.assertTrue(data["sessions"][0]["is_active"])

    def test_get_session_history(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.get("/sessions/s1")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], "s1")
        self.assertEqual(len(data["messages"]), 2)
        self.assertEqual(data["messages"][0]["role"], "user")

    def test_switch_session(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.post("/sessions/switch/s2")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["session_id"], "s2")
        agent.switch_session.assert_called_once_with("s2")

    def test_reset_session(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.post("/sessions/reset")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        agent.reset_session.assert_called_once()

    def test_export_session(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.get("/sessions/s1/export")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("会话导出", resp.text)
        agent.export_session_markdown.assert_called_once_with("s1")

    def test_stats_endpoint(self) -> None:
        agent = _make_agent()
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.get("/stats")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("summary", data)
        self.assertIn("recent", data)


if __name__ == "__main__":
    unittest.main()
