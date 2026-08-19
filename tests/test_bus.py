"""事件总线测试。"""

from __future__ import annotations

import asyncio
import unittest

from bus import EventBus, TurnCommitted


class EventBusTest(unittest.IsolatedAsyncioTestCase):
    async def test_emit_calls_handlers_in_order(self) -> None:
        bus = EventBus()
        calls: list[str] = []

        async def handler_a(event: TurnCommitted) -> None:
            calls.append("a")

        async def handler_b(event: TurnCommitted) -> None:
            calls.append("b")

        bus.on("turn_committed", handler_a)
        bus.on("turn_committed", handler_b)
        await bus.emit(TurnCommitted(user_input="hi", assistant_reply="hello"))

        self.assertEqual(calls, ["a", "b"])

    async def test_emit_handler_can_replace_event(self) -> None:
        bus = EventBus()

        async def enrich(event: TurnCommitted) -> TurnCommitted:
            return TurnCommitted(
                user_input=event.user_input,
                assistant_reply=event.assistant_reply + "!",
            )

        async def observe(event: TurnCommitted) -> None:
            observe.seen = event.assistant_reply

        bus.on("turn_committed", enrich)
        bus.on("turn_committed", observe)
        result = await bus.emit(
            TurnCommitted(user_input="hi", assistant_reply="hello")
        )

        self.assertEqual(result.assistant_reply, "hello!")
        self.assertEqual(observe.seen, "hello!")

    async def test_fanout_runs_concurrently_and_collects_failures(self) -> None:
        bus = EventBus()

        async def slow_ok(event: TurnCommitted) -> None:
            await asyncio.sleep(0.01)

        async def boom(event: TurnCommitted) -> None:
            raise ValueError("爆炸")

        bus.on("turn_committed", slow_ok)
        bus.on("turn_committed", boom)
        await bus.fanout(TurnCommitted())

        self.assertEqual(len(bus.failures), 1)
        self.assertIn("爆炸", bus.failures[0][1])

    async def test_enqueue_processes_in_background(self) -> None:
        bus = EventBus()
        bus.start()
        processed: list[str] = []

        async def handler(event: TurnCommitted) -> None:
            processed.append(event.user_input)

        bus.on("turn_committed", handler)
        await bus.enqueue(TurnCommitted(user_input="第一轮"))
        await bus.enqueue(TurnCommitted(user_input="第二轮"))
        await bus.drain()

        self.assertEqual(processed, ["第一轮", "第二轮"])
        await bus.aclose()

    async def test_global_observer_receives_all_events(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        async def watcher(event: TurnCommitted) -> None:
            seen.append(event.type)

        bus.on_any(watcher)
        await bus.emit(TurnCommitted())

        self.assertEqual(seen, ["turn_committed"])


if __name__ == "__main__":
    unittest.main()
