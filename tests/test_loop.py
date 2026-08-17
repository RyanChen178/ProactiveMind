"""Agent ReAct 循环测试。"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from agent.loop import AgentLoop
from agent.provider import LLMResponse, StreamEvent, ToolCall
from agent.session import Session


class FakeProvider:
    """按顺序返回预设模型响应。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return next(self._responses)


class FakeTools:
    """记录工具调用并返回预设结果。"""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self._results = results or {}
        self.calls: list[ToolCall] = []

    def get_schemas(self) -> list[dict]:
        return []

    async def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        return self._results.get(call.name, "ok")


class FakeStreamProvider:
    """逐段返回预设文本的流式 Provider。"""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[list[dict]] = []

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ):
        self.calls.append(messages)
        for chunk in self._chunks:
            yield StreamEvent(content=chunk)
        yield StreamEvent(response=LLMResponse(content="".join(self._chunks)))


class FakeConsolidator:
    """记录后台归档请求。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def consolidate(self, user_input: str, assistant_reply: str) -> list[str]:
        self.calls.append((user_input, assistant_reply))
        return []


def build_loop(
    responses: list[LLMResponse],
    tool_results: dict[str, str] | None = None,
) -> tuple[AgentLoop, FakeProvider, FakeTools]:
    """构建不访问网络和文件的 AgentLoop。"""

    loop = AgentLoop.__new__(AgentLoop)
    provider = FakeProvider(responses)
    tools = FakeTools(tool_results)
    loop._provider = provider
    loop._session = Session()
    loop._config = SimpleNamespace(max_history_tokens=6000)
    loop._system_prompt = "system"
    loop._tools = tools
    return loop, provider, tools


class AgentLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_streams_reply_and_persists_final_message(self) -> None:
        loop, _, _ = build_loop([])
        loop._provider = FakeStreamProvider(["你", "好"])

        chunks = [chunk async for chunk in loop.run_stream("在吗")]

        self.assertEqual(chunks, ["你", "好"])
        self.assertEqual(loop._provider.calls[0][-1]["content"], "在吗")
        self.assertEqual(
            [message["role"] for message in loop._session.messages],
            ["user", "assistant"],
        )
        self.assertEqual(loop._session.messages[-1]["content"], "你好")

    async def test_returns_direct_reply(self) -> None:
        loop, provider, tools = build_loop([LLMResponse(content="你好")])

        reply = await loop.run("在吗？")

        self.assertEqual(reply, "你好")
        self.assertEqual(
            [message["role"] for message in provider.calls[0]],
            ["system", "user"],
        )
        self.assertEqual(
            [message["role"] for message in loop._session.messages],
            ["user", "assistant"],
        )
        self.assertEqual(tools.calls, [])

    async def test_schedules_consolidation_after_final_reply(self) -> None:
        loop, _, _ = build_loop([LLMResponse(content="你好")])
        consolidator = FakeConsolidator()
        loop._config = SimpleNamespace(
            max_history_tokens=6000,
            consolidation=SimpleNamespace(enabled=True),
        )
        loop._consolidator = consolidator
        loop._consolidation_tasks = set()

        reply = await loop.run("在吗？")
        await asyncio.sleep(0)

        self.assertEqual(reply, "你好")
        self.assertEqual(consolidator.calls, [("在吗？", "你好")])

    async def test_places_assistant_tool_call_before_tool_result(self) -> None:
        responses = [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("call-time", "get_time", {})],
            ),
            LLMResponse(content="现在是十点。"),
        ]
        loop, provider, tools = build_loop(
            responses,
            {"get_time": "2026-08-09 10:00:00"},
        )

        reply = await loop.run("现在几点？")

        self.assertEqual(reply, "现在是十点。")
        self.assertEqual(
            [message["role"] for message in provider.calls[1]],
            ["system", "user", "assistant", "tool"],
        )
        assistant_call = provider.calls[1][2]["tool_calls"][0]
        self.assertEqual(assistant_call["id"], "call-time")
        self.assertEqual(assistant_call["function"]["arguments"], "{}")
        self.assertEqual(provider.calls[1][3]["tool_call_id"], "call-time")
        self.assertEqual([call.name for call in tools.calls], ["get_time"])

    async def test_keeps_multiple_tool_results_in_one_call_group(self) -> None:
        responses = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("call-time", "get_time", {}),
                    ToolCall("call-memory", "recall", {"keyword": "项目"}),
                ],
            ),
            LLMResponse(content="工具都执行完成。"),
        ]
        loop, provider, tools = build_loop(
            responses,
            {"get_time": "10:00", "recall": "ProactiveMind"},
        )

        reply = await loop.run("查看时间和项目记忆")

        self.assertEqual(reply, "工具都执行完成。")
        self.assertEqual(
            [message["role"] for message in provider.calls[1]],
            ["system", "user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in provider.calls[1][3:]],
            ["call-time", "call-memory"],
        )
        self.assertEqual(
            [call.name for call in tools.calls],
            ["get_time", "recall"],
        )

    async def test_records_message_when_max_steps_is_reached(self) -> None:
        response = LLMResponse(
            content="",
            tool_calls=[ToolCall("call-time", "get_time", {})],
        )
        loop, _, _ = build_loop([response])

        reply = await loop.run("一直调用工具", max_steps=1)

        self.assertEqual(reply, "（达到最大工具调用次数，终止本轮）")
        self.assertEqual(
            [message["role"] for message in loop._session.messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(loop._session.messages[-1]["content"], reply)


if __name__ == "__main__":
    unittest.main()
