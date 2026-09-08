"""ProactiveMind SDK 客户端实现。

底层协议：JSON-RPC 2.0 over NDJSON（每行一个 JSON 对象）。
传输：asyncio.StreamReader/Writer，支持 TCP 主机:端口 或 Unix 域 socket。

用法：
  async with await AsyncProactiveMind.connect("127.0.0.1:6324") as client:
      session = await client.session_start()
      async for event in client.turn_start(session["session_id"], "你好"):
          ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator, Iterator
from concurrent.futures import Future
from typing import Any, cast
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
PROTOCOL_VERSION = "1.0"
CLIENT_NAME = "proactivemind-sdk"
CLIENT_VERSION = "0.1.0"


class RemoteError(RuntimeError):
    """服务端返回的错误响应。"""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        self.retryable = isinstance(data, dict) and bool(data.get("retryable"))


class ConnectionClosedError(ConnectionError):
    """连接已关闭。"""

    pass


class ProtocolError(RuntimeError):
    """协议帧解析错误。"""

    pass


class SlowConsumerError(ConnectionError):
    """订阅消费方处理速度跟不上服务端推送速度。"""

    pass


class SessionSubscription:
    """订阅一个会话的事件流。"""

    def __init__(
        self,
        wire: "_WireClient",
        session_id: str,
        queue_size: int,
    ) -> None:
        self._wire = wire
        self.session_id = session_id
        self.id = uuid4().hex
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(queue_size)
        self._closed = False
        self._error: Exception | None = None
        self._iterator_lock = threading.Lock()
        self._iterator_claimed = False

    def _finish(self, error: Exception | None = None) -> None:
        """标记订阅为关闭态，并把 None 哨兵推入队列让消费者跳出循环。"""
        if self._closed:
            return
        self._closed = True
        self._error = error
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(None)

    async def events(self) -> AsyncGenerator[dict[str, Any], None]:
        """异步迭代事件。订阅生命周期内重复调用会抛错。"""
        with self._iterator_lock:
            if self._iterator_claimed:
                raise RuntimeError(
                    "subscription already has a consumer; re-subscribe from saved seq"
                )
            self._iterator_claimed = True
        while not self._closed:
            event = await self._queue.get()
            if event is None:
                break
            yield event
        if self._error is not None:
            raise self._error

    async def close(self) -> None:
        self._finish()
        await self._wire._unfollow(self)

    async def __aenter__(self) -> "SessionSubscription":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()


class _WireClient:
    """底层连接：StreamReader/Writer 之上做 JSON-RPC 2.0 帧解析。"""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.pending: dict[int, asyncio.Future[object]] = {}
        self.subscriptions: dict[str, SessionSubscription] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self._follow_lock = asyncio.Lock()
        self.next_id = 1
        self.closed = False
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="proactivemind-sdk-reader"
        )

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> "_WireClient":
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        if endpoint.count(":") == 1 and not endpoint.startswith("/"):
            host, raw_port = endpoint.rsplit(":", 1)
            try:
                port = int(raw_port)
            except ValueError as exc:
                raise ValueError(f"invalid port: {raw_port}") from exc
            reader, writer = await asyncio.open_connection(
                host, port, limit=max_message_bytes + 1
            )
        else:
            reader, writer = await asyncio.open_unix_connection(
                endpoint, limit=max_message_bytes + 1
            )
        wire = cls(reader, writer)
        try:
            await wire.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "version": CLIENT_VERSION,
                },
                "workspaceToken": workspace_token,
            })
            await wire.notify("initialized", {})
        except BaseException:
            await wire.close()
            raise
        return wire

    async def request(self, method: str, params: dict[str, object]) -> object:
        if self.closed:
            raise ConnectionClosedError("connection is closed")
        request_id = self.next_id
        self.next_id += 1
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._write({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
            return await future
        except BaseException:
            future.cancel()
            raise

    async def notify(self, method: str, params: dict[str, object]) -> None:
        await self._write({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })

    async def _write(self, payload: dict[str, object]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self.writer.write(encoded)
        await self.writer.drain()

    async def _unfollow(self, subscription: SessionSubscription) -> None:
        if self.subscriptions.pop(subscription.id, None) is None or self.closed:
            return
        try:
            await self.request("session/unfollow", {
                "session_id": subscription.session_id,
                "subscription_id": subscription.id,
            })
        except ConnectionClosedError:
            pass

    def _stop_later(self, subscription: SessionSubscription) -> None:
        """异步后台清理订阅：发送 unfollow，错误只记录日志。"""
        task = asyncio.create_task(
            self._unfollow(subscription), name="proactivemind-sdk-unfollow"
        )
        self.tasks.add(task)

        def _on_done(task: asyncio.Task[None]) -> None:
            self.tasks.discard(task)
            if not task.cancelled() and (exc := task.exception()) is not None:
                logger.error("session unfollow failed", exc_info=exc)

        task.add_done_callback(_on_done)

    async def follow(
        self,
        session_id: str,
        after_seq: int,
        queue_size: int,
    ) -> SessionSubscription:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        async with self._follow_lock:
            subscription = SessionSubscription(self, session_id, queue_size)
            self.subscriptions[subscription.id] = subscription
            try:
                await self.request("session/follow", {
                    "session_id": session_id,
                    "after_seq": after_seq,
                    "subscription_id": subscription.id,
                })
            except BaseException:
                subscription._finish()
                self._stop_later(subscription)
                raise
            # ACK 后清理同会话的旧订阅，避免事件混到旧队列。
            for previous in tuple(self.subscriptions.values()):
                if (
                    previous.session_id == session_id
                    and previous is not subscription
                ):
                    previous._finish()
                    self.subscriptions.pop(previous.id, None)
            return subscription

    async def _read_loop(self) -> None:
        """单 reader 分发响应与通知，慢订阅仅关闭自身。"""
        failure: Exception = ConnectionClosedError("server closed connection")
        try:
            while line := await self.reader.readline():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"invalid JSON frame: {exc}") from exc
                if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
                    raise ProtocolError("JSON-RPC frame must be a version 2.0 object")
                message = cast(dict[str, Any], value)
                if "id" in message:
                    await self._handle_response(message)
                else:
                    await self._handle_notification(message)
                await asyncio.sleep(0)  # 让消费者有机会调度
        except asyncio.CancelledError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            failure = ProtocolError(f"invalid server frame: {exc}")
        except Exception as exc:
            failure = exc
        finally:
            self._dispatch_failure(failure)

    async def _handle_response(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            raise ProtocolError("response id must be int")
        future = self.pending.pop(request_id, None)
        if future is None:
            raise ProtocolError(f"unknown response id: {request_id}")
        if future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            future.set_exception(RemoteError(
                int(error["code"]),
                str(error["message"]),
                error.get("data"),
            ))
        else:
            future.set_result(message.get("result"))

    async def _handle_notification(self, message: dict[str, Any]) -> None:
        if message.get("method") not in ("session/event", "session/error"):
            raise ProtocolError("unknown server notification")
        params = message.get("params")
        if not isinstance(params, dict):
            raise ProtocolError("notification params must be object")
        identity = params.get("subscription_id")
        if not isinstance(identity, str):
            raise ProtocolError("subscription_id must be str")
        subscription = self.subscriptions.get(identity)
        if subscription is None or subscription._closed:
            return
        if message["method"] == "session/error":
            error = params.get("error") or {}
            subscription._finish(RemoteError(
                -32603,
                str(error.get("message", "remote error")),
                error,
            ))
            self._stop_later(subscription)
            return
        event = params.get("event")
        if (
            not isinstance(event, dict)
            or event.get("session_id") != subscription.session_id
        ):
            raise ProtocolError("session event does not match subscription")
        try:
            subscription._queue.put_nowait(event)
        except asyncio.QueueFull:
            subscription._finish(SlowConsumerError(
                f"session queue overflow; re-subscribe from saved seq: {subscription.session_id}"
            ))
            self._stop_later(subscription)

    def _dispatch_failure(self, failure: Exception) -> None:
        """reader 终止时通知所有等待者。"""
        self.closed = True
        # 取消所有 pending 请求
        for future in self.pending.values():
            if not future.done():
                future.set_exception(failure)
        self.pending.clear()
        # 关闭所有订阅
        for subscription in self.subscriptions.values():
            subscription._finish(failure)
        self.subscriptions.clear()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            # 唤醒所有等待者
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(
                        ConnectionClosedError("connection closed by client")
                    )
            self.pending.clear()


class AsyncProactiveMind:
    """异步 SDK 客户端。"""

    def __init__(self, wire: _WireClient) -> None:
        self._wire = wire

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> "AsyncProactiveMind":
        wire = await _WireClient.connect(
            endpoint,
            workspace_token=workspace_token,
            max_message_bytes=max_message_bytes,
        )
        return cls(wire)

    async def request(self, method: str, params: dict[str, object]) -> object:
        return await self._wire.request(method, params)

    async def session_start(self) -> dict[str, Any]:
        """创建新会话，返回 { session_id, created_at }。"""
        result = await self._wire.request("session/start", {})
        if not isinstance(result, dict):
            raise ProtocolError("session/start result must be object")
        return result

    async def session_list(
        self,
        *,
        cursor: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, object] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        result = await self._wire.request("session/list", params)
        if not isinstance(result, dict):
            raise ProtocolError("session/list result must be object")
        return result

    async def session_follow(
        self,
        session_id: str,
        *,
        after_seq: int = -1,
        queue_size: int = 256,
    ) -> SessionSubscription:
        return await self._wire.follow(session_id, after_seq, queue_size)

    async def session_switch(self, session_id: str) -> dict[str, Any]:
        result = await self._wire.request("session/switch", {"session_id": session_id})
        if not isinstance(result, dict):
            raise ProtocolError("session/switch result must be object")
        return result

    async def turn_start(
        self,
        session_id: str,
        text: str,
        *,
        message_id: str | None = None,
        queue_size: int = 256,
    ) -> SessionSubscription:
        """发起一轮 turn，并返回事件订阅。

        服务端会通过 session/event 推送 delta / tool_call / turn_done 等。
        调用方应迭代 subscription.events() 消费事件。
        """
        params: dict[str, object] = {
            "session_id": session_id,
            "text": text,
            "message_id": message_id or uuid4().hex,
        }
        subscription = await self._wire.follow(session_id, -1, queue_size)
        await self._wire.request("turn/start", params)
        return subscription

    async def turn_interrupt(self, session_id: str, message_id: str | None = None) -> None:
        params: dict[str, object] = {"session_id": session_id}
        if message_id is not None:
            params["message_id"] = message_id
        await self._wire.request("turn/interrupt", params)

    async def close(self) -> None:
        await self._wire.close()

    async def __aenter__(self) -> "AsyncProactiveMind":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()


class ProactiveMind:
    """同步 SDK 客户端。

    通过后台线程持有 asyncio 事件循环，异步方法在此事件循环中执行。
    """

    def __init__(self, async_client: AsyncProactiveMind) -> None:
        self._async = async_client

    @classmethod
    def connect(
        cls,
        endpoint: str,
        *,
        workspace_token: str | None = None,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> "ProactiveMind":
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, daemon=True, name="proactivemind-sdk-loop"
        )
        thread.start()
        future: Future[AsyncProactiveMind] = asyncio.run_coroutine_threadsafe(
            AsyncProactiveMind.connect(
                endpoint,
                workspace_token=workspace_token,
                max_message_bytes=max_message_bytes,
            ),
            loop,
        )
        try:
            client = future.result()
        except BaseException:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1.0)
            loop.close()
            raise
        sync = cls.__new__(cls)
        sync._async = client
        sync._loop = loop
        sync._thread = thread
        return sync

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def session_start(self) -> dict[str, Any]:
        return self._run(self._async.session_start())

    def session_list(
        self,
        *,
        cursor: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return self._run(self._async.session_list(cursor=cursor, limit=limit))

    def session_switch(self, session_id: str) -> dict[str, Any]:
        return self._run(self._async.session_switch(session_id))

    def turn_start(self, session_id: str, text: str) -> "SyncTurnIterator":
        """发起 turn 并返回同步事件迭代器。"""

        async def _open() -> tuple[SessionSubscription, AsyncGenerator[dict[str, Any], None]]:
            sub = await self._async.turn_start(session_id, text)
            return sub, sub.events()

        sub, async_iter = self._run(_open())
        return SyncTurnIterator(self._loop, sub, async_iter)

    def turn_interrupt(self, session_id: str, message_id: str | None = None) -> None:
        self._run(self._async.turn_interrupt(session_id, message_id))

    def close(self) -> None:
        try:
            self._run(self._async.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)
            self._loop.close()

    def __enter__(self) -> "ProactiveMind":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class SyncTurnIterator:
    """同步 turn 事件迭代器：在同步 SDK 中暴露异步订阅流。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        subscription: SessionSubscription,
        async_iter: AsyncGenerator[dict[str, Any], None],
    ) -> None:
        self._loop = loop
        self._sub = subscription
        self._iter = async_iter

    def __iter__(self) -> "SyncTurnIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        try:
            return asyncio.run_coroutine_threadsafe(
                self._iter.__anext__(), self._loop
            ).result()
        except StopAsyncIteration:
            raise StopIteration

    def close(self) -> None:
        future = asyncio.run_coroutine_threadsafe(self._sub.close(), self._loop)
        future.result(timeout=5.0)