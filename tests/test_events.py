"""事件总线测试。"""

from __future__ import annotations

import asyncio
import unittest

from events import EventHub, TurnCompleted


class EventHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_emit_calls_handlers_in_order(self) -> None:
        bus = EventHub()
        calls: list[str] = []

        async def handler_a(event: TurnCompleted) -> None:
            calls.append("a")

        async def handler_b(event: TurnCompleted) -> None:
            calls.append("b")

        bus.on("turn_committed", handler_a)
        bus.on("turn_committed", handler_b)
        await bus.emit(TurnCompleted(user_input="hi", assistant_reply="hello"))

        self.assertEqual(calls, ["a", "b"])

    async def test_emit_handler_can_replace_event(self) -> None:
        bus = EventHub()

        async def enrich(event: TurnCompleted) -> TurnCompleted:
            return TurnCompleted(
                user_input=event.user_input,
                assistant_reply=event.assistant_reply + "!",
            )

        async def observe(event: TurnCompleted) -> None:
            observe.seen = event.assistant_reply

        bus.on("turn_committed", enrich)
        bus.on("turn_committed", observe)
        result = await bus.emit(
            TurnCompleted(user_input="hi", assistant_reply="hello")
        )

        self.assertEqual(result.assistant_reply, "hello!")
        self.assertEqual(observe.seen, "hello!")

    async def test_fanout_runs_concurrently_and_collects_failures(self) -> None:
        bus = EventHub()

        async def slow_ok(event: TurnCompleted) -> None:
            await asyncio.sleep(0.01)

        async def boom(event: TurnCompleted) -> None:
            raise ValueError("爆炸")

        bus.on("turn_committed", slow_ok)
        bus.on("turn_committed", boom)
        await bus.fanout(TurnCompleted())

        self.assertEqual(len(bus.failures), 1)
        self.assertIn("爆炸", bus.failures[0][1])

    async def test_enqueue_processes_in_background(self) -> None:
        bus = EventHub()
        bus.start()
        processed: list[str] = []

        async def handler(event: TurnCompleted) -> None:
            processed.append(event.user_input)

        bus.on("turn_committed", handler)
        await bus.enqueue(TurnCompleted(user_input="第一轮"))
        await bus.enqueue(TurnCompleted(user_input="第二轮"))
        await bus.drain()

        self.assertEqual(processed, ["第一轮", "第二轮"])
        await bus.aclose()

    async def test_global_observer_receives_all_events(self) -> None:
        bus = EventHub()
        seen: list[str] = []

        async def watcher(event: TurnCompleted) -> None:
            seen.append(event.type)

        bus.on_any(watcher)
        await bus.emit(TurnCompleted())

        self.assertEqual(seen, ["turn_committed"])


if __name__ == "__main__":
    unittest.main()
