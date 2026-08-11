"""上下文预算与历史投影。"""

from __future__ import annotations

import json
import math


def estimate_message_tokens(message: dict) -> int:
    """估算一条 OpenAI 格式消息占用的 token 数。"""

    text_parts = [str(message.get("content") or "")]
    if "tool_calls" in message:
        text_parts.append(
            json.dumps(message["tool_calls"], ensure_ascii=False, separators=(",", ":"))
        )
    if "tool_call_id" in message:
        text_parts.append(str(message["tool_call_id"]))
    return 4 + math.ceil(len("".join(text_parts)) / 4)


def build_history_view(messages: list[dict], max_tokens: int) -> list[dict]:
    """按预算保留最近的完整消息组，不改写原始历史。"""

    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")

    groups = _group_messages(messages)
    selected: list[list[dict]] = []
    used_tokens = 0

    for group in reversed(groups):
        group_tokens = sum(estimate_message_tokens(message) for message in group)
        if selected and used_tokens + group_tokens > max_tokens:
            break
        selected.append(group)
        used_tokens += group_tokens

    return [message for group in reversed(selected) for message in group]


def _group_messages(messages: list[dict]) -> list[list[dict]]:
    """将 assistant 工具调用及其结果归为同一组。"""

    groups: list[list[dict]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1

        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1

        groups.append(group)
    return groups
