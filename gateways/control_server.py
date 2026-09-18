"""Agent Control Protocol 服务端 —— JSON-RPC 2.0 over TCP (NDJSON)。

协议方法（与 sdk/python 客户端对齐）：
  请求：
    initialize           握手，返回 serverInfo
    session/start        新建会话，返回 {session_id, created_at}
    session/list         列出会话（支持 limit/cursor 占位）
    session/switch       切换活动会话
    session/follow       订阅会话事件流（带 subscription_id）
    session/unfollow     取消订阅
    turn/start           发起一轮 turn，立即返回 ack；事件经 session/event 推送
    turn/interrupt       中断当前 turn
  通知（客户端 -> 服务端，无 id）：
    initialized          握手完成
  通知（服务端 -> 客户端）：
    session/event        {subscription_id, event:{session_id, seq, type, ...}}
    session/error        {subscription_id, error:{code, message}}

事件类型（event.type）：
  turn_started / delta / turn_done / turn_error
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from mind.loop import MindLoop, TurnInterruptedError

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
SERVER_NAME = "proactivemind-control"
SERVER_VERSION = "0.1.0"

MAX_FRAME_BYTES = 2 * 1024 * 1024
TURN_BUSY_CODE = -32000
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


@dataclass
class _Subscription:
    """一个客户端订阅。"""

    subscription_id: str
    session_id: str
    writer: asyncio.StreamWriter

    def event_frame(self, event: dict[str, Any]) -> bytes:
        payload = {
            "jsonrpc": "2.0",
            "method": "session/event",
            "params": {
                "subscription_id": self.subscription_id,
                "event": event,
            },
        }
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    def error_frame(self, code: int, message: str) -> bytes:
        payload = {
            "jsonrpc": "2.0",
            "method": "session/error",
            "params": {
                "subscription_id": self.subscription_id,
                "error": {"code": code, "message": message},
            },
        }
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


class _ClientSession:
    """单个 TCP 连接的状态。"""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.initialized = False
        self.subscriptions: dict[str, _Subscription] = {}
        self._write_lock = asyncio.Lock()

    async def send(self, frame: bytes) -> None:
        """串行写帧，避免并发写交错。连接断开时静默。"""
        async with self._write_lock:
            try:
                self.writer.write(frame)
                await self.writer.drain()
            except (ConnectionError, BrokenPipeError, RuntimeError):
                pass

    async def send_response(self, request_id: int, result: Any) -> None:
        frame = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        await self.send((json.dumps(frame, ensure_ascii=False) + "\n").encode())

    async def send_error(self, request_id: int, code: int, message: str) -> None:
        frame = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        await self.send((json.dumps(frame, ensure_ascii=False) + "\n").encode())


def _err_response(request_id: int, code: int, message: str) -> bytes:
    frame = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    return (json.dumps(frame, ensure_ascii=False) + "\n").encode()


class ControlServer:
    """TCP 控制服务：每连接一个 _ClientSession，全局共享 turn 执行。"""

    def __init__(self, agent: MindLoop, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self._agent = agent
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[_ClientSession] = set()
        self._turn_lock = asyncio.Lock()
        self._turn_task: asyncio.Task | None = None
        # 每个 session 的事件序号（订阅者按 seq 断点续读）
        self._seqs: dict[str, int] = {}

    @property
    def port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self._port
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> int:
        """启动 TCP 监听，返回实际端口。"""
        self._server = await asyncio.start_server(
            self._handle_client, self._host, self._port, limit=MAX_FRAME_BYTES + 1
        )
        log.info(
            "ControlServer 监听 %s:%d",
            self._host,
            self.port,
        )
        return self.port

    async def stop(self) -> None:
        """停止监听并断开所有客户端，取消进行中的 turn。"""
        if self._turn_task is not None and not self._turn_task.done():
            self._agent.interrupt_current("server shutdown")
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for client in list(self._clients):
            client.writer.close()
        self._clients.clear()

    # ------------------------------------------------------------------
    # 连接处理
    # ------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        session = _ClientSession(writer)
        self._clients.add(session)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                if len(line) > MAX_FRAME_BYTES:
                    await session.send(_err_response(
                        0, PARSE_ERROR, "frame exceeds size limit",
                    ))
                    continue
                await self._dispatch_line(session, line)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self._clients.discard(session)
            session.subscriptions.clear()
            try:
                writer.close()
            except Exception:
                pass

    async def _dispatch_line(self, session: _ClientSession, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            await session.send(_err_response(0, PARSE_ERROR, "invalid JSON"))
            return
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            request_id = msg.get("id") if isinstance(msg, dict) else 0
            await session.send(_err_response(
                request_id if isinstance(request_id, int) else 0,
                INVALID_REQUEST,
                "expected JSON-RPC 2.0 object",
            ))
            return

        method = msg.get("method")
        params = msg.get("params") or {}
        has_id = "id" in msg
        request_id = msg.get("id")

        if not isinstance(method, str):
            if has_id:
                await session.send_error(request_id, INVALID_REQUEST, "method required")
            return

        handler = getattr(self, f"_rpc_{method.replace('/', '_')}", None)
        if handler is None:
            if has_id:
                await session.send_error(
                    request_id, METHOD_NOT_FOUND, f"unknown method: {method}"
                )
            return

        # initialize 之外的请求需要先握手
        if method != "initialize" and not session.initialized:
            if has_id:
                await session.send_error(
                    request_id, INVALID_REQUEST, "initialize required first"
                )
            return

        try:
            if has_id:
                result = await handler(session, params)
                await session.send_response(request_id, result)
            else:
                await handler(session, params)
        except _ParamError as exc:
            if has_id:
                await session.send_error(request_id, INVALID_PARAMS, str(exc))
        except _BusyError as exc:
            if has_id:
                await session.send_error(request_id, TURN_BUSY_CODE, str(exc))
        except Exception as exc:
            log.exception("control method %s failed", method)
            if has_id:
                await session.send_error(request_id, -32603, f"internal error: {exc}")

    # ------------------------------------------------------------------
    # RPC 方法
    # ------------------------------------------------------------------

    async def _rpc_initialize(self, session: _ClientSession, params: dict) -> dict:
        # initialize 成功即视为握手完成；initialized 通知可选（兼容两类客户端）
        session.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"turnStreaming": True},
        }

    async def _rpc_initialized(self, session: _ClientSession, params: dict) -> None:
        session.initialized = True

    async def _rpc_session_start(self, session: _ClientSession, params: dict) -> dict:
        self._agent.reset_session()
        return {
            "session_id": self._agent._session_id,
            "created_at": _now_iso(),
        }

    async def _rpc_session_list(self, session: _ClientSession, params: dict) -> dict:
        limit = params.get("limit", 50)
        if not isinstance(limit, int) or limit <= 0:
            raise _ParamError("limit must be a positive integer")
        items = self._agent._session_store.list_sessions()[:limit]
        return {
            "sessions": [
                {
                    "session_id": it.get("id"),
                    "created_at": it.get("created_at"),
                    "message_count": it.get("message_count"),
                    "is_active": it.get("is_active"),
                }
                for it in items
            ],
            "next_cursor": None,
        }

    async def _rpc_session_switch(self, session: _ClientSession, params: dict) -> dict:
        session_id = params.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise _ParamError("session_id must be a non-empty string")
        if not self._agent.switch_session(session_id):
            raise _ParamError(f"unknown session_id: {session_id}")
        return {"session_id": session_id}

    async def _rpc_session_follow(self, session: _ClientSession, params: dict) -> dict:
        session_id = params.get("session_id")
        subscription_id = params.get("subscription_id")
        if not isinstance(session_id, str) or not session_id:
            raise _ParamError("session_id must be a non-empty string")
        if not isinstance(subscription_id, str) or not subscription_id:
            raise _ParamError("subscription_id must be a non-empty string")
        # 替换同会话旧订阅（与 SDK 客户端语义一致）
        for sub_id, sub in list(session.subscriptions.items()):
            if sub.session_id == session_id and sub_id != subscription_id:
                del session.subscriptions[sub_id]
        session.subscriptions[subscription_id] = _Subscription(
            subscription_id=subscription_id,
            session_id=session_id,
            writer=session.writer,
        )
        return {"subscription_id": subscription_id}

    async def _rpc_session_unfollow(self, session: _ClientSession, params: dict) -> dict:
        subscription_id = params.get("subscription_id")
        removed = session.subscriptions.pop(subscription_id, None)
        return {"removed": removed is not None}

    async def _rpc_turn_start(self, session: _ClientSession, params: dict) -> dict:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _ParamError("text must be a non-empty string")
        message_id = params.get("message_id") or uuid4().hex
        requested_session = params.get("session_id") or self._agent._session_id

        async with self._turn_lock:
            if self._turn_task is not None and not self._turn_task.done():
                raise _BusyError("a turn is already running")
            self._turn_task = asyncio.create_task(
                self._run_turn(requested_session, text, message_id),
                name=f"control-turn-{message_id}",
            )
        return {"message_id": message_id, "session_id": requested_session}

    async def _rpc_turn_interrupt(self, session: _ClientSession, params: dict) -> dict:
        interrupted = self._agent.interrupt_current("control protocol")
        return {"interrupted": interrupted}

    # ------------------------------------------------------------------
    # turn 执行与事件推送
    # ------------------------------------------------------------------

    def _next_seq(self, session_id: str) -> int:
        self._seqs[session_id] = self._seqs.get(session_id, 0) + 1
        return self._seqs[session_id]

    async def _broadcast_event(self, session_id: str, event: dict[str, Any]) -> None:
        """把事件推给所有客户端中订阅了该会话的订阅者。"""
        event = {"session_id": session_id, **event}
        for client in list(self._clients):
            for sub in list(client.subscriptions.values()):
                if sub.session_id != session_id:
                    continue
                await client.send(sub.event_frame(event))

    async def _broadcast_error(self, session_id: str, message: str) -> None:
        for client in list(self._clients):
            for sub in list(client.subscriptions.values()):
                if sub.session_id != session_id:
                    continue
                await client.send(sub.error_frame(-32000, message))

    async def _run_turn(self, session_id: str, text: str, message_id: str) -> None:
        """执行一轮 turn，把流式输出转成事件广播。"""
        try:
            await self._broadcast_event(session_id, {
                "seq": self._next_seq(session_id),
                "type": "turn_started",
                "message_id": message_id,
            })
            started = time.monotonic()
            chunks: list[str] = []
            async for piece in self._agent.run_stream(text):
                chunks.append(piece)
                await self._broadcast_event(session_id, {
                    "seq": self._next_seq(session_id),
                    "type": "delta",
                    "message_id": message_id,
                    "content": piece,
                })
            await self._broadcast_event(session_id, {
                "seq": self._next_seq(session_id),
                "type": "turn_done",
                "message_id": message_id,
                "content": "".join(chunks),
                "duration_ms": round((time.monotonic() - started) * 1000, 2),
            })
        except TurnInterruptedError:
            await self._broadcast_event(session_id, {
                "seq": self._next_seq(session_id),
                "type": "turn_done",
                "message_id": message_id,
                "reason": "interrupted",
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("control turn failed")
            await self._broadcast_event(session_id, {
                "seq": self._next_seq(session_id),
                "type": "turn_error",
                "message_id": message_id,
                "error": str(exc),
            })
            await self._broadcast_error(session_id, f"turn failed: {exc}")


class _ParamError(ValueError):
    pass


class _BusyError(RuntimeError):
    pass


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def serve_control(
    agent: MindLoop, *, host: str = "127.0.0.1", port: int = 6324
) -> None:
    """便捷入口：启动 ControlServer 并阻塞直到关闭。"""
    server = ControlServer(agent, host=host, port=port)
    bound = await server.start()
    print(f"ProactiveMind Control — {host}:{bound}")
    try:
        await asyncio.Event().wait()  # 阻塞直到取消
    finally:
        await server.stop()