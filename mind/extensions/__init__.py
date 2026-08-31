"""扩展系统模块

提供生命周期钩子、事件总线、工具拦截器和热重载功能。
"""

from mind.extensions.lifecycle import (
    LifecycleHook,
    LifecycleHooks,
    TurnContext,
    get_lifecycle_hooks,
    register_hook,
    unregister_hook,
    invoke_hooks,
)

from mind.extensions.events import (
    EventBus,
    EventHandler,
    # Event data classes
    TurnStartedData,
    TurnCompletedData,
    ToolExecutedData,
    MemoryChangedData,
    ExtensionLoadedData,
    ExtensionUnloadedData,
    SessionCreatedData,
    SessionSwitchedData,
    ErrorOccurredData,
    # Event classes
    TurnStartedEvent,
    TurnCompletedEvent,
    ToolExecutedEvent,
    MemoryChangedEvent,
    ExtensionLoadedEvent,
    ExtensionUnloadedEvent,
    SessionCreatedEvent,
    SessionSwitchedEvent,
    ErrorOccurredEvent,
    get_event_bus,
    # Emit functions
    emit_turn_started,
    emit_turn_completed,
    emit_tool_executed,
    emit_memory_changed,
    emit_extension_loaded,
    emit_extension_unloaded,
    emit_session_created,
    emit_session_switched,
    emit_error_occurred,
)

from mind.extensions.interceptor import (
    ToolInterceptor,
    ToolInterceptorResult,
    ToolInterceptorRegistry,
    get_tool_interceptor_registry,
    register_global_interceptor,
    register_tool_interceptor,
    unregister_global_interceptor,
    unregister_tool_interceptor,
    intercept_tool_call,
    on_tool_pre,
    is_tool_interceptor,
    get_intercept_tool,
)

from mind.extensions.hot_reload import (
    ExtensionFileState,
    GenerationInfo,
    HotReloader,
    get_hot_reloader,
    init_hot_reloader,
    stop_hot_reloader,
)

__all__ = [
    # Lifecycle
    "LifecycleHook",
    "LifecycleHooks",
    "TurnContext",
    "get_lifecycle_hooks",
    "register_hook",
    "unregister_hook",
    "invoke_hooks",
    
    # Events
    "EventBus",
    "EventHandler",
    "TurnStartedData",
    "TurnCompletedData",
    "ToolExecutedData",
    "MemoryChangedData",
    "ExtensionLoadedData",
    "ExtensionUnloadedData",
    "SessionCreatedData",
    "SessionSwitchedData",
    "ErrorOccurredData",
    "TurnStartedEvent",
    "TurnCompletedEvent",
    "ToolExecutedEvent",
    "MemoryChangedEvent",
    "ExtensionLoadedEvent",
    "ExtensionUnloadedEvent",
    "SessionCreatedEvent",
    "SessionSwitchedEvent",
    "ErrorOccurredEvent",
    "get_event_bus",
    "emit_turn_started",
    "emit_turn_completed",
    "emit_tool_executed",
    "emit_memory_changed",
    "emit_extension_loaded",
    "emit_extension_unloaded",
    "emit_session_created",
    "emit_session_switched",
    "emit_error_occurred",
    
    # Interceptor
    "ToolInterceptor",
    "ToolInterceptorResult",
    "ToolInterceptorRegistry",
    "get_tool_interceptor_registry",
    "register_global_interceptor",
    "register_tool_interceptor",
    "unregister_global_interceptor",
    "unregister_tool_interceptor",
    "intercept_tool_call",
    "on_tool_pre",
    "is_tool_interceptor",
    "get_intercept_tool",
    
    # Hot Reload
    "ExtensionFileState",
    "GenerationInfo",
    "HotReloader",
    "get_hot_reloader",
    "init_hot_reloader",
    "stop_hot_reloader",
]
