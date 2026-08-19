"""事件总线 —— 组件间解耦的事件系统。

提供三种语义：
  emit   —— 干预链：handler 按顺序执行，返回值非 None 时替换事件
  fanout —— 并发观察：所有 handler 并发执行，单个失败不打断
  enqueue —— 后台异步：入队后由 dispatcher 异步 fanout
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

EventHandler = Callable[[Any], Awaitable[Any | None]]


@dataclass
class Event:
    """事件基类。"""

    type: str


@dataclass
class TurnCommitted(Event):
    """一轮对话完成提交。"""

    type: str = "turn_committed"
    session_id: str = ""
    user_input: str = ""
    assistant_reply: str = ""


@dataclass
class MemoryWritten(Event):
    """记忆文件写入完成。"""

    type: str = "memory_written"
    target: str = ""
    fact: str = ""


class EventBus:
    """异步事件总线。"""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global: list[EventHandler] = []
        self._queue: asyncio.Queue[Event] | None = None
        self._dispatcher: asyncio.Task | None = None
        self._failures: list[tuple[str, str]] = []

    def on(self, event_type: str, handler: EventHandler) -> None:
        """注册某个事件类型的 handler。"""
        self._handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """注册全局观察者。"""
        self._global.append(handler)

    async def emit(self, event: Event) -> Event:
        """干预链：按注册顺序执行，handler 返回非 None 时替换事件。"""

        for handler in list(self._handlers.get(event.type, [])):
            result = await handler(event)
            if result is not None:
                event = result
        for handler in list(self._global):
            await handler(event)
        return event

    async def fanout(self, event: Event) -> None:
        """并发观察：所有 handler 并发执行，汇总失败。"""

        handlers = list(self._handlers.get(event.type, [])) + list(self._global)
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(event) for h in handlers), return_exceptions=True
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handler_name = getattr(handlers[i], "__name__", str(handlers[i]))
                error_msg = f"{event.type}/{handler_name}: {result}"
                self._failures.append((event.type, str(result)))
                log.warning("fanout handler 失败: %s", error_msg)

    def start(self) -> None:
        """启动后台 dispatcher。"""
        if self._dispatcher is not None:
            return
        self._queue = asyncio.Queue()
        self._dispatcher = asyncio.create_task(self._run_dispatcher())

    async def enqueue(self, event: Event) -> None:
        """入队后由 dispatcher 异步 fanout。"""

        if self._queue is None:
            await self.fanout(event)
            return
        await self._queue.put(event)

    async def _run_dispatcher(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self.fanout(event)
            except Exception:
                log.exception("dispatcher 处理 %s 时崩溃", getattr(event, "type", "?"))
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        """等待已入队事件处理完成。"""
        if self._queue is not None:
            await self._queue.join()

    @property
    def failures(self) -> list[tuple[str, str]]:
        return list(self._failures)

    async def aclose(self) -> None:
        """关闭：停止 dispatcher。"""
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except asyncio.CancelledError:
                pass
            self._dispatcher = None
        self._queue = None
