"""扩展系统测试"""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from mind.extensions.lifecycle import (
    get_lifecycle_hooks,
    TurnContext,
    register_hook,
    unregister_hook,
    invoke_hooks,
)
from mind.extensions.events import (
    get_event_bus,
    TurnStartedEvent,
    TurnStartedData,
    emit_turn_started,
)
from mind.extensions.interceptor import (
    get_tool_interceptor_registry,
    ToolInterceptorResult,
    register_global_interceptor,
    intercept_tool_call,
    on_tool_pre,
    is_tool_interceptor,
)

class TestLifecycleHooks(unittest.TestCase):
    def setUp(self):
        get_lifecycle_hooks().clear()

    def test_register_and_invoke(self):
        hooks = get_lifecycle_hooks()
        called = []
        def my_hook(ctx):
            called.append(True)
            return ctx
        hooks.register("before_turn", my_hook)
        ctx = TurnContext(session_id="test", user_message="hello")
        asyncio.run(hooks.invoke("before_turn", ctx))
        self.assertEqual(len(called), 1)

    def test_multiple_hooks(self):
        hooks = get_lifecycle_hooks()
        order = []
        def hook1(ctx):
            order.append(1)
            return ctx
        def hook2(ctx):
            order.append(2)
            return ctx
        hooks.register("before_turn", hook1)
        hooks.register("before_turn", hook2)
        ctx = TurnContext(session_id="test", user_message="hello")
        asyncio.run(hooks.invoke("before_turn", ctx))
        self.assertEqual(order, [1, 2])

class TestEventBus(unittest.TestCase):
    def setUp(self):
        get_event_bus().clear()

    def test_subscribe_and_publish(self):
        bus = get_event_bus()
        received = []
        async def handler(event):
            received.append(event)
        bus.subscribe("turn_started", handler)
        event = TurnStartedEvent(name="turn_started", data=TurnStartedData(session_id="test", user_message="hello"))
        asyncio.run(bus.publish(event))
        self.assertEqual(len(received), 1)

    def test_emit_convenience(self):
        bus = get_event_bus()
        received = []
        async def handler(event):
            received.append(event)
        bus.subscribe_all(handler)
        asyncio.run(emit_turn_started("test", "hello"))
        self.assertEqual(len(received), 1)

class TestToolInterceptor(unittest.TestCase):
    def setUp(self):
        get_tool_interceptor_registry().clear()

    def test_global_interceptor(self):
        registry = get_tool_interceptor_registry()
        called = []
        async def interceptor(tool_name, arguments):
            called.append(tool_name)
            return ToolInterceptorResult()
        registry.register_global(interceptor)
        asyncio.run(registry.intercept("shell", {}))
        self.assertEqual(called, ["shell"])

    def test_block_interceptor(self):
        registry = get_tool_interceptor_registry()
        async def blocker(tool_name, arguments):
            return ToolInterceptorResult(should_block=True, block_reason="blocked")
        registry.register_global(blocker)
        result = asyncio.run(registry.intercept("shell", {}))
        self.assertTrue(result.should_block)

    def test_decorator(self):
        @on_tool_pre("shell")
        async def my_interceptor(tool_name, arguments):
            return ToolInterceptorResult()
        self.assertTrue(is_tool_interceptor(my_interceptor))

if __name__ == "__main__":
    unittest.main()
