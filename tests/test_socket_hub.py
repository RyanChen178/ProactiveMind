"""SocketHub 测试。"""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from gateways.web_chat import SocketHub


class SocketHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_sends_to_all_connections(self) -> None:
        cm = SocketHub()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        cm._connections = [ws1, ws2]

        await cm.broadcast("hello")

        expected = json.dumps({"type": "proactive", "content": "hello"})
        ws1.send_text.assert_awaited_once_with(expected)
        ws2.send_text.assert_awaited_once_with(expected)

    async def test_broadcast_removes_dead_connections(self) -> None:
        cm = SocketHub()
        ws_alive = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_text.side_effect = RuntimeError("disconnected")
        cm._connections = [ws_alive, ws_dead]

        await cm.broadcast("ping")

        ws_alive.send_text.assert_awaited_once()
        self.assertEqual(cm.count, 1)
        self.assertIn(ws_alive, cm._connections)

    async def test_broadcast_empty_does_nothing(self) -> None:
        cm = SocketHub()
        await cm.broadcast("test")
        self.assertEqual(cm.count, 0)

    def test_disconnect_removes_connection(self) -> None:
        cm = SocketHub()
        ws = MagicMock()
        cm._connections = [ws]

        cm.disconnect(ws)

        self.assertEqual(cm.count, 0)

    def test_disconnect_ignores_unknown_connection(self) -> None:
        cm = SocketHub()
        ws = MagicMock()
        cm.disconnect(ws)
        self.assertEqual(cm.count, 0)


if __name__ == "__main__":
    unittest.main()
