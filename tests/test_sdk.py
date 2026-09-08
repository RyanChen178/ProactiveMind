"""ProactiveMind Python SDK 测试。

测试策略：用 asyncio.start_server 启动一个 in-process NDJSON echo server，
根据收到的请求回写 JSON-RPC 帧驱动客户端。run() 协程负责消费所有请求并触发推送。
"""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from proactivemind_sdk import (
    AsyncProactiveMind,
    ConnectionClosedError,
    ProtocolError,
    RemoteError,
)
from proactivemind_sdk.client import _WireClient


class _FakeServer:
    """轻量级 NDJSON 服务端：根据客户端请求动态响应。

    用法：
      server = _FakeServer()
      server.scripted = [(request_method, response_dict), ...]
      # 顺序匹配请求方法，未匹配的请求回 _ok_response(id, {})
      await server.start()
      try:
          ...  # 测试主体
      finally:
          await server.stop()
    """

    def __init__(self) -> None:
        # list[(method, response_dict)] 顺序消费
        self.scripted: list[tuple[str, dict[str, Any]]] = []
        self.received: list[dict[str, Any]] = []
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        self.host: str = "127.0.0.1"
        self.port: int = 0
        self._ready = asyncio.Event()
        self._close_event = asyncio.Event()
        # 推送帧：test 通过 server.push() 主动推送
        self._push_handlers: list[Any] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_client, self.host, 0
        )
        sock = self._server.sockets[0].getsockname()
        self.port = sock[1]

    async def stop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    async def push(self, frame: dict[str, Any]) -> None:
        """主动推送一个 JSON-RPC 帧（给订阅用）。"""
        if self._writer is None:
            raise RuntimeError("no client connected")
        payload = (json.dumps(frame, ensure_ascii=False) + "\n").encode()
        self._writer.write(payload)
        await self._writer.drain()

    async def _on_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writer = writer
        self._ready.set()
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    break
                self.received.append(msg)
                await self._handle_request(msg, writer)
        except (asyncio.CancelledError, ConnectionError, ValueError):
            pass
        finally:
            self._close_event.set()

    async def _handle_request(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        method = msg.get("method")
        request_id = msg.get("id")
        # 匹配 scripted 响应（按方法名顺序消费）
        for i, (m, resp) in enumerate(self.scripted):
            if m == method:
                self.scripted.pop(i)
                # 用真实 request_id 覆盖 id 字段，保留 result/error
                resp = dict(resp)
                if request_id is not None:
                    resp["id"] = request_id
                writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())
                await writer.drain()
                return
        # 默认 OK 响应
        if request_id is not None:
            writer.write(
                (json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {}}, ensure_ascii=False) + "\n").encode()
            )
            await writer.drain()


def _ok_response(request_id: int, result: Any = None) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result or {}}


def _err_response(request_id: int, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class HandshakeTest(unittest.IsolatedAsyncioTestCase):
    """握手流程：initialize + initialized。"""

    async def test_successful_handshake(self) -> None:
        server = _FakeServer()
        # initialize 用 _ok_response；initialized 是 notify，不应被响应
        server.scripted = [("initialize", _ok_response(0, {"serverInfo": {"name": "test", "version": "0.0.1"}}))]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            await client.close()
            self.assertEqual(len(server.received), 2)
            self.assertEqual(server.received[0]["method"], "initialize")
            self.assertEqual(server.received[1]["method"], "initialized")
            self.assertEqual(
                server.received[0]["params"]["clientInfo"]["name"],
                "proactivemind-sdk",
            )
        finally:
            await server.stop()

    async def test_unix_socket_endpoint(self) -> None:
        import sys

        if sys.platform == "win32":
            self.skipTest("Unix socket unavailable on Windows")
        import tempfile

        sock_path = tempfile.mktemp(prefix="pm-sdk-", suffix=".sock")
        server = _FakeServer()
        server.scripted = [("initialize", _ok_response(0, {}))]
        server._server = await asyncio.start_unix_server(server._on_client, sock_path)
        try:
            client = await AsyncProactiveMind.connect(sock_path)
            await client.close()
        finally:
            await server.stop()
            import os as _os

            try:
                _os.unlink(sock_path)
            except OSError:
                pass


class RequestResponseTest(unittest.IsolatedAsyncioTestCase):
    """普通 request/response 与错误传播。"""

    async def test_request_returns_result(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/start", _ok_response(0, {"session_id": "abc123"})),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            result = await client.session_start()
            self.assertEqual(result["session_id"], "abc123")
            await client.close()
        finally:
            await server.stop()

    async def test_request_raises_remote_error(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/start", _err_response(0, -32600, "invalid request")),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            with self.assertRaises(RemoteError) as ctx:
                await client.session_start()
            self.assertEqual(ctx.exception.code, -32600)
            await client.close()
        finally:
            await server.stop()


class SessionListTest(unittest.IsolatedAsyncioTestCase):
    """session/list 分页参数。"""

    async def test_session_list_with_cursor(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/list", _ok_response(0, {"sessions": [], "next_cursor": None})),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            result = await client.session_list(cursor=["abc"], limit=10)
            self.assertIn("sessions", result)
            self.assertGreaterEqual(len(server.received), 3)
            # received[0]=initialize, [1]=initialized, [2]=session/list
            sent = server.received[2]
            self.assertEqual(sent["method"], "session/list")
            self.assertEqual(sent["params"]["cursor"], ["abc"])
            self.assertEqual(sent["params"]["limit"], 10)
            await client.close()
        finally:
            await server.stop()


class SubscriptionTest(unittest.IsolatedAsyncioTestCase):
    """session/event 与订阅生命周期。"""

    async def test_subscription_receives_events(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/follow", _ok_response(0, {})),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            sub = await client.session_follow("sess-1")

            # 后台任务推送 2 条事件
            async def _push_events() -> None:
                # 等 follow 完成
                await asyncio.sleep(0.01)
                await server.push({
                    "jsonrpc": "2.0",
                    "method": "session/event",
                    "params": {
                        "subscription_id": sub.id,
                        "event": {
                            "session_id": "sess-1",
                            "seq": 1,
                            "type": "delta",
                            "content": "你好",
                        },
                    },
                })
                await server.push({
                    "jsonrpc": "2.0",
                    "method": "session/event",
                    "params": {
                        "subscription_id": sub.id,
                        "event": {
                            "session_id": "sess-1",
                            "seq": 2,
                            "type": "turn_done",
                        },
                    },
                })

            push_task = asyncio.create_task(_push_events())

            events = []
            async for event in sub.events():
                events.append(event)
                if event.get("type") == "turn_done":
                    await sub.close()
                    break
            await push_task
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["type"], "delta")
            self.assertEqual(events[0]["content"], "你好")
            self.assertEqual(events[1]["type"], "turn_done")
            await client.close()
        finally:
            await server.stop()


class TurnStartTest(unittest.IsolatedAsyncioTestCase):
    """turn/start 订阅 + 事件流。"""

    async def test_turn_start_subscribes_and_receives_stream(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/follow", _ok_response(0, {})),
            ("turn/start", _ok_response(0, {"message_id": "m1"})),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            sub = await client.turn_start("sess-2", "ping", message_id="m1")

            async def _push_events() -> None:
                await asyncio.sleep(0.01)
                for seq, content in [(1, "Hello "), (2, "world")]:
                    await server.push({
                        "jsonrpc": "2.0",
                        "method": "session/event",
                        "params": {
                            "subscription_id": sub.id,
                            "event": {
                                "session_id": "sess-2",
                                "seq": seq,
                                "type": "delta",
                                "content": content,
                            },
                        },
                    })
                await server.push({
                    "jsonrpc": "2.0",
                    "method": "session/event",
                    "params": {
                        "subscription_id": sub.id,
                        "event": {
                            "session_id": "sess-2",
                            "seq": 3,
                            "type": "turn_done",
                        },
                    },
                })

            push_task = asyncio.create_task(_push_events())

            chunks: list[str] = []
            async for event in sub.events():
                if event.get("type") == "turn_done":
                    await sub.close()
                    break
                if event.get("type") == "delta":
                    chunks.append(event["content"])
            await push_task
            self.assertEqual("".join(chunks), "Hello world")
            await client.close()
        finally:
            await server.stop()


class ErrorPropagationTest(unittest.IsolatedAsyncioTestCase):
    """服务端错误通知。"""

    async def test_session_error_finishes_subscription(self) -> None:
        server = _FakeServer()
        server.scripted = [
            ("initialize", _ok_response(0, {})),
            ("session/follow", _ok_response(0, {})),
        ]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            sub = await client.session_follow("sess-err")

            async def _push_error() -> None:
                await asyncio.sleep(0.01)
                await server.push({
                    "jsonrpc": "2.0",
                    "method": "session/error",
                    "params": {
                        "subscription_id": sub.id,
                        "error": {"code": -32603, "message": "session closed"},
                    },
                })

            push_task = asyncio.create_task(_push_error())

            with self.assertRaises(RemoteError):
                async for _ in sub.events():
                    pass
            await push_task
            await client.close()
        finally:
            await server.stop()


class ConnectionFailureTest(unittest.IsolatedAsyncioTestCase):
    """连接关闭时的错误传播。"""

    async def test_pending_requests_fail_when_server_closes(self) -> None:
        server = _FakeServer()
        server.scripted = [("initialize", _ok_response(0, {}))]
        await server.start()
        try:
            client = await AsyncProactiveMind.connect(server.endpoint())
            # 服务端 close writer 后客户端请求应失败
            assert server._writer is not None
            server._writer.close()
            try:
                await server._writer.wait_closed()
            except Exception:
                pass
            # 给客户端 reader 一点时间检测 EOF
            await asyncio.sleep(0.05)
            with self.assertRaises(ConnectionClosedError):
                await client.session_start()
        finally:
            await server.stop()


class InvalidFrameTest(unittest.IsolatedAsyncioTestCase):
    """协议帧错误处理。"""

    async def test_invalid_json_raises_protocol_error(self) -> None:
        server = _FakeServer()
        await server.start()
        try:
            async def _send_garbage() -> None:
                await server._ready.wait()
                if server._writer is not None:
                    server._writer.write(b"not json\n")
                    await server._writer.drain()

            asyncio.create_task(_send_garbage())
            with self.assertRaises(ProtocolError):
                await AsyncProactiveMind.connect(server.endpoint())
        finally:
            await server.stop()


if __name__ == "__main__":
    unittest.main()