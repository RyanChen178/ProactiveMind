"""LLM Provider —— OpenAI Chat Completions 兼容调用。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from mind.config import LLMConfig


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 一次调用的结果。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """流式响应中的文本增量或最终完整响应。"""

    content: str = ""
    response: LLMResponse | None = None


class LLMProvider:
    """OpenAI Chat Completions 兼容的 LLM 调用客户端。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """调用 chat completions 接口。"""

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""

        tool_calls: list[ToolCall] = []
        for raw_tc in choice.get("tool_calls") or []:
            func = raw_tc["function"]
            try:
                arguments = json.loads(func["arguments"])
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=raw_tc["id"], name=func["name"], arguments=arguments)
            )

        usage = data.get("usage", {})
        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """调用 SSE 流式接口，并在结束时给出完整响应。"""

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        usage: dict[str, int] = {}
        completed = False
        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw_data = line[6:]
                if raw_data == "[DONE]":
                    completed = True
                    break
                data = json.loads(raw_data)
                usage = data.get("usage") or usage
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    content_parts.append(content)
                    yield StreamEvent(content=content)
                self._merge_tool_call_deltas(tool_calls, delta.get("tool_calls") or [])

        if not completed:
            raise RuntimeError("LLM 流式响应未正常结束")
        yield StreamEvent(
            response=LLMResponse(
                content="".join(content_parts),
                tool_calls=self._parse_stream_tool_calls(tool_calls),
                usage=usage,
            )
        )

    @staticmethod
    def _merge_tool_call_deltas(
        target: dict[int, dict[str, str]], deltas: list[dict[str, Any]]
    ) -> None:
        """合并 SSE 中被拆开的工具调用字段。"""

        for delta in deltas:
            index = delta["index"]
            call = target.setdefault(index, {"id": "", "name": "", "arguments": ""})
            call["id"] += delta.get("id") or ""
            function = delta.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""

    @staticmethod
    def _parse_stream_tool_calls(
        raw_calls: dict[int, dict[str, str]]
    ) -> list[ToolCall]:
        """将合并后的工具调用转换为内部表示。"""

        calls = []
        for _, raw_call in sorted(raw_calls.items()):
            try:
                arguments = json.loads(raw_call["arguments"])
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=raw_call["id"],
                    name=raw_call["name"],
                    arguments=arguments,
                )
            )
        return calls

    async def aclose(self) -> None:
        await self._client.aclose()
