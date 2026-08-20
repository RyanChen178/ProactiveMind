"""Web Chat 渠道测试。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from channels.web_chat import create_app


def _make_agent(stream_chunks: list[str]) -> MagicMock:
    """构建模拟 Agent，run_stream 产出指定文本块。"""

    async def _run_stream(user_input: str, max_steps: int = 10):
        for chunk in stream_chunks:
            yield chunk

    agent = MagicMock()
    agent.run_stream = _run_stream
    return agent


class WebChatTest(unittest.TestCase):
    def test_index_returns_html(self) -> None:
        agent = _make_agent([])
        app = create_app(agent)

        with TestClient(app) as client:
            resp = client.get("/")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("ProactiveMind", resp.text)
        self.assertIn("WebSocket", resp.text)

    def test_websocket_receives_delta_and_done(self) -> None:
        agent = _make_agent(["你好", "，世界"])
        app = create_app(agent)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "message", "content": "hi"})
                messages = []
                while True:
                    data = ws.receive_json()
                    messages.append(data)
                    if data.get("type") == "done":
                        break

        deltas = [m["content"] for m in messages if m["type"] == "delta"]
        self.assertEqual(deltas, ["你好", "，世界"])
        self.assertEqual(messages[-1]["type"], "done")

    def test_websocket_ignores_non_message(self) -> None:
        agent = _make_agent(["ok"])
        app = create_app(agent)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "ping"})
                ws.send_json({"type": "message", "content": "hello"})
                messages = []
                while True:
                    data = ws.receive_json()
                    messages.append(data)
                    if data.get("type") == "done":
                        break

        deltas = [m["content"] for m in messages if m["type"] == "delta"]
        self.assertEqual(deltas, ["ok"])

    def test_websocket_ignores_empty_content(self) -> None:
        agent = _make_agent(["resp"])
        app = create_app(agent)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "message", "content": "   "})
                ws.send_json({"type": "message", "content": "real"})
                messages = []
                while True:
                    data = ws.receive_json()
                    messages.append(data)
                    if data.get("type") == "done":
                        break

        deltas = [m["content"] for m in messages if m["type"] == "delta"]
        self.assertEqual(deltas, ["resp"])


if __name__ == "__main__":
    unittest.main()
