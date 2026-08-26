"""会话管理 —— 内存中的消息历史。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from mind.context import build_history_view


MessagePersister = Callable[[dict], None]


@dataclass
class Session:
    """一次对话会话，保存完整的消息历史。

    messages 采用 OpenAI 格式：
      {"role": "user"|"assistant"|"tool", "content": "...", ...}
    """

    messages: list[dict] = field(default_factory=list)
    persist_message: MessagePersister | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def add_user(self, content: str) -> None:
        self._append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def _append(self, message: dict) -> None:
        """先持久化消息，再更新当前内存视图。"""

        if self.persist_message is not None:
            self.persist_message(message)
        self.messages.append(message)

    def get_history(self, max_tokens: int = 6000) -> list[dict]:
        """返回受预算约束的完整消息组视图。"""

        return build_history_view(self.messages, max_tokens)

    def clear(self) -> None:
        self.messages.clear()
