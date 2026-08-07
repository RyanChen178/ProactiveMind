"""会话管理 —— 内存中的消息历史。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    """一次对话会话，保存完整的消息历史。

    messages 采用 OpenAI 格式：
      {"role": "user"|"assistant"|"tool", "content": "...", ...}
    """

    messages: list[dict] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def get_history(self, max_messages: int = 50) -> list[dict]:
        """返回最近的消息历史，避免超出上下文窗口。"""
        if len(self.messages) <= max_messages:
            return list(self.messages)
        return list(self.messages[-max_messages:])

    def clear(self) -> None:
        self.messages.clear()
