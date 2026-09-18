"""Agent Control Protocol 服务端测试。

用真实 TCP + asyncio 驱动，并使用 sdk/python 客户端做端到端验证。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from gateways.control_server import ControlServer
from mind.provider import LLMResponse, StreamEvent

# SDK 客户端（PYTHONPATH 已包含 sdk/python/src）
from proactivemind_sdk import AsyncProactiveMind, RemoteError


class _FakeProvider:
    async def chat(self, messages, tools=None):
        return LLMResponse(content="ok", tool_calls=[], usage={})

    async def chat_stream(self, messages, tools=None):
        for piece in ["你", "好"]:
            await asyncio.sleep(0)
            yield StreamEvent(content=piece)
        yield StreamEvent(
            content="",
            response=LLMResponse(content="你好", tool_calls=[], usage={}),
        )


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent._session_id = "sess-1"
    agent._session_store.list_sessions.return_value = [
        {"id": "sess-1", "created_at": "2026-01-01", "message_count": 3, "is_active": True},
        {"id": "sess-2", "created_at": "2026-01-02", "message_count": 1, "is_active": False},
    ]
    provider = _FakeProvider()
    agent._provider = provider
    agent.run_stream = _fake_run_stream
    agent.switch_session = MagicMock(side_effect=lambda sid: sid in {"sess-1", "sess-2"})
    agent.interrupt_current = MagicMock(return_value=False)
    return agent


async def _fake_run_stream(text: str, max_steps: int = 10):
    for piece in ["你", "好"]:
        await asyncio.sleep(0.005)
        yield piece


class ControlServerFrameTest(unittest.IsolatedAsyncioTestCase):
    """直接 TCP 帧级测试。"""

    async def _start(self) -> tuple[ControlServer, asyncio.StreamReader, asyncio.StreamWriter]:
        agent = _make_agent()
        server = ControlServer(agent, host="127.0.0.1", port=0)
        port = await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        self.addCleanup(writer.close)
        self.addCleanup(server.stop)
        return server, reader, writer

    async def _roundtrip(self, writer, reader, payload: dict) -> dict:
        writer.write((json.dumps(payload) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        return json.loads(line)

    async def test_initialize_handshake(self) -> None:
        _, reader, writer = await self._start()
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["protocolVersion"], "1.0")
        self.assertEqual(resp["result"]["serverInfo"]["name"], "proactivemind-control")

    async def test_invalid_json_returns_parse_error(self) -> None:
        _, reader, writer = await self._start()
        writer.write(b"not json\n")
        await writer.drain()
        resp = json.loads(await asyncio.wait_for(reader.readline(), timeout=2.0))
        self.assertEqual(resp["error"]["code"], -32700)

    async def test_unknown_method(self) -> None:
        _, reader, writer = await self._start()
        # 先握手
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "no/such", "params": {},
        })
        self.assertEqual(resp["error"]["code"], -32601)

    async def test_request_before_initialize_rejected(self) -> None:
        _, reader, writer = await self._start()
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 5, "method": "session/list", "params": {},
        })
        self.assertEqual(resp["error"]["code"], -32600)
        self.assertIn("initialize", resp["error"]["message"])

    async def test_session_list(self) -> None:
        _, reader, writer = await self._start()
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "session/list", "params": {"limit": 10},
        })
        sessions = resp["result"]["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["session_id"], "sess-1")

    async def test_session_start_resets(self) -> None:
        agent = _make_agent()
        server = ControlServer(agent, host="127.0.0.1", port=0)
        port = await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        self.addCleanup(writer.close)
        self.addCleanup(server.stop)

        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "session/start", "params": {},
        })
        self.assertEqual(resp["result"]["session_id"], "sess-1")
        self.assertIn("created_at", resp["result"])
        agent.reset_session.assert_called_once()

    async def test_invalid_params_reported(self) -> None:
        _, reader, writer = await self._start()
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "turn/start", "params": {"text": "  "},
        })
        self.assertEqual(resp["error"]["code"], -32602)

    async def test_turn_events_pushed_to_follower(self) -> None:
        _, reader, writer = await self._start()
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "initialized", "params": {},
        })
        # 订阅
        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 3,
            "method": "session/follow",
            "params": {"session_id": "sess-1", "subscription_id": "sub-1"},
        })
        # 发起 turn（ack）
        ack = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 4,
            "method": "turn/start",
            "params": {"session_id": "sess-1", "text": "hi", "message_id": "m1"},
        })
        self.assertEqual(ack["result"]["message_id"], "m1")

        # 读取推送的事件直到 turn_done
        events = []
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            msg = json.loads(line)
            self.assertEqual(msg["method"], "session/event")
            event = msg["params"]["event"]
            self.assertEqual(msg["params"]["subscription_id"], "sub-1")
            events.append(event)
            if event["type"] == "turn_done":
                break

        types = [e["type"] for e in events]
        self.assertEqual(types[0], "turn_started")
        self.assertIn("delta", types)
        self.assertEqual(types[-1], "turn_done")
        # seq 单调递增
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, sorted(seqs))

    async def test_turn_interrupt(self) -> None:
        agent = _make_agent()
        server = ControlServer(agent, host="127.0.0.1", port=0)
        port = await server.start()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        self.addCleanup(writer.close)
        self.addCleanup(server.stop)

        await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        resp = await self._roundtrip(writer, reader, {
            "jsonrpc": "2.0", "id": 2, "method": "turn/interrupt", "params": {},
        })
        agent.interrupt_current.assert_called_once()
        self.assertIn("interrupted", resp["result"])


class SdkEndToEndTest(unittest.IsolatedAsyncioTestCase):
    """用 SDK 客户端连真实服务端做端到端验证。"""

    async def test_sdk_full_flow(self) -> None:
        agent = _make_agent()
        server = ControlServer(agent, host="127.0.0.1", port=0)
        port = await server.start()
        self.addCleanup(server.stop)

        client = await AsyncProactiveMind.connect(f"127.0.0.1:{port}")
        try:
            # 会话列表
            listing = await client.session_list(limit=10)
            self.assertEqual(len(listing["sessions"]), 2)

            # 订阅并发起 turn，收齐事件流
            sub = await client.turn_start("sess-1", "你好", message_id="sdk-m1")
            try:
                chunks: list[str] = []
                async for event in sub.events():
                    if event["type"] == "turn_done":
                        break
                    if event["type"] == "delta":
                        chunks.append(event["content"])
                self.assertEqual("".join(chunks), "你好")
            finally:
                await sub.close()
        finally:
            await client.close()

    async def test_sdk_remote_error_propagates(self) -> None:
        agent = _make_agent()
        server = ControlServer(agent, host="127.0.0.1", port=0)
        port = await server.start()
        self.addCleanup(server.stop)

        client = await AsyncProactiveMind.connect(f"127.0.0.1:{port}")
        try:
            with self.assertRaises(RemoteError) as ctx:
                await client.session_switch("no-such-session")
            # 服务端把 unknown session 报为 invalid params
            self.assertEqual(ctx.exception.code, -32602)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()