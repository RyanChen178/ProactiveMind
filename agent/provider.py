"""LLM Provider —— OpenAI Chat Completions 兼容调用。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.config import LLMConfig


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
    ) -> LLMResponse:
        """调用 chat completions 接口。"""

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
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

    async def aclose(self) -> None:
        await self._client.aclose()
