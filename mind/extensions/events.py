"""扩展系统事件总线

支持 9 种事件类型：
- turn_started: Turn 开始
- turn_completed: Turn 完成
- tool_executed: 工具执行完成
- memory_changed: 记忆发生变化
- extension_loaded: 扩展加载
- extension_unloaded: 扩展卸载
- session_created: 会话创建
- session_switched: 会话切换
- error_occurred: 错误发生
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

# 事件处理器类型
EventHandler = Callable[[Any], Any | Awaitable[Any]]


@dataclass
class TurnStartedData:
    """Turn 开始事件数据"""
    session_id: str
    user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnCompletedData:
    """Turn 完成事件数据"""
    session_id: str
    user_message: str
    response: str
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class ToolExecutedData:
    """工具执行完成事件数据"""
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    success: bool


@dataclass
class MemoryChangedData:
    """记忆变化事件数据"""
    change_type: str  # "add", "update", "delete"
    key: str
    value: Any = None


@dataclass
class ExtensionLoadedData:
    """扩展加载事件数据"""
    extension_name: str
    module_path: str


@dataclass
class ExtensionUnloadedData:
    """扩展卸载事件数据"""
    extension_name: str


@dataclass
class SessionCreatedData:
    """会话创建事件数据"""
    session_id: str


@dataclass
class SessionSwitchedData:
    """会话切换事件数据"""
    from_session_id: str
    to_session_id: str


@dataclass
class ErrorOccurredData:
    """错误发生事件数据"""
    error_type: str
    error_message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnStartedEvent:
    """Turn 开始事件"""
    name: str
    data: TurnStartedData


@dataclass
class TurnCompletedEvent:
    """Turn 完成事件"""
    name: str
    data: TurnCompletedData


@dataclass
class ToolExecutedEvent:
    """工具执行完成事件"""
    name: str
    data: ToolExecutedData


@dataclass
class MemoryChangedEvent:
    """记忆变化事件"""
    name: str
    data: MemoryChangedData


@dataclass
class ExtensionLoadedEvent:
    """扩展加载事件"""
    name: str
    data: ExtensionLoadedData


@dataclass
class ExtensionUnloadedEvent:
    """扩展卸载事件"""
    name: str
    data: ExtensionUnloadedData


@dataclass
class SessionCreatedEvent:
    """会话创建事件"""
    name: str
    data: SessionCreatedData


@dataclass
class SessionSwitchedEvent:
    """会话切换事件"""
    name: str
    data: SessionSwitchedData


@dataclass
class ErrorOccurredEvent:
    """错误发生事件"""
    name: str
    data: ErrorOccurredData


class EventBus:
    """事件总线，用于发布和订阅事件"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = {}
            cls._instance._global_handlers = []
        return cls._instance
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """订阅特定类型的事件"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """订阅所有事件"""
        self._global_handlers.append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """取消订阅特定类型的事件"""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
    
    def unsubscribe_all(self, handler: EventHandler) -> None:
        """取消订阅所有事件"""
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)
    
    def clear(self) -> None:
        """清空所有事件处理器"""
        self._handlers.clear()
        self._global_handlers.clear()
    
    async def publish(self, event: Any) -> None:
        """发布事件"""
        event_type = event.name if hasattr(event, 'name') else type(event).__name__
        
        # 调用特定类型的处理器
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Error in event handler for {event_type}: {e}"
                )
        
        # 调用全局处理器
        for handler in self._global_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Error in global event handler: {e}"
                )


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    return EventBus()


# 便捷函数：发布事件
async def emit_turn_started(session_id: str, user_message: str, metadata: dict[str, Any] | None = None) -> None:
    """发布 Turn 开始事件"""
    event = TurnStartedEvent(
        name="turn_started",
        data=TurnStartedData(session_id, user_message, metadata or {})
    )
    await get_event_bus().publish(event)


async def emit_turn_completed(session_id: str, user_message: str, response: str, tool_calls: list[str] | None = None) -> None:
    """发布 Turn 完成事件"""
    event = TurnCompletedEvent(
        name="turn_completed",
        data=TurnCompletedData(session_id, user_message, response, tool_calls or [])
    )
    await get_event_bus().publish(event)


async def emit_tool_executed(tool_name: str, arguments: dict[str, Any], result: Any, success: bool) -> None:
    """发布工具执行完成事件"""
    event = ToolExecutedEvent(
        name="tool_executed",
        data=ToolExecutedData(tool_name, arguments, result, success)
    )
    await get_event_bus().publish(event)


async def emit_memory_changed(change_type: str, key: str, value: Any = None) -> None:
    """发布记忆变化事件"""
    event = MemoryChangedEvent(
        name="memory_changed",
        data=MemoryChangedData(change_type, key, value)
    )
    await get_event_bus().publish(event)


async def emit_extension_loaded(extension_name: str, module_path: str) -> None:
    """发布扩展加载事件"""
    event = ExtensionLoadedEvent(
        name="extension_loaded",
        data=ExtensionLoadedData(extension_name, module_path)
    )
    await get_event_bus().publish(event)


async def emit_extension_unloaded(extension_name: str) -> None:
    """发布扩展卸载事件"""
    event = ExtensionUnloadedEvent(
        name="extension_unloaded",
        data=ExtensionUnloadedData(extension_name)
    )
    await get_event_bus().publish(event)


async def emit_session_created(session_id: str) -> None:
    """发布会话创建事件"""
    event = SessionCreatedEvent(
        name="session_created",
        data=SessionCreatedData(session_id)
    )
    await get_event_bus().publish(event)


async def emit_session_switched(from_session_id: str, to_session_id: str) -> None:
    """发布会话切换事件"""
    event = SessionSwitchedEvent(
        name="session_switched",
        data=SessionSwitchedData(from_session_id, to_session_id)
    )
    await get_event_bus().publish(event)


async def emit_error_occurred(error_type: str, error_message: str, context: dict[str, Any] | None = None) -> None:
    """发布错误发生事件"""
    event = ErrorOccurredEvent(
        name="error_occurred",
        data=ErrorOccurredData(error_type, error_message, context or {})
    )
    await get_event_bus().publish(event)
