"""主动推送去重 —— 判断新消息是否与近期推送过雷同。

设计思想（与 a20b45af proactive_v2/judge.py 中 MessageDeduper 对齐）：

  - 把近期 N 条 proactive 消息 + 新消息 + 状态标签喂给 LLM，
    让它返回 JSON：{is_duplicate: bool, reason: str}
  - 失败容错：LLM 调用异常时按"非重复"放行，避免误判阻塞推送
  - 配套的 _format_recent_proactive_entries 把消息列表序列化为
    "1) ...\n---\n2) ..." 格式
  - _recent_meta / _field 是字段提取工具，兼容 dict 与对象两种形态

ProactiveMind 命名（initiative/proactive_deduper.py）独立于原仓库。
不依赖具体 LLMProvider 抽象，只要求 .chat() 接口返回带 .content 字段的对象。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol

from initiative.json_extract import extract_json_object

logger = logging.getLogger(__name__)

DUPE_MAX_TOKENS_CAP = 128
PLACEHOLDER_REASON = ""


class _ChatLike(Protocol):
    """最小 LLM 协议：chat() 返回带 .content 字段的对象。"""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = ...,
        model: str | None = ...,
        max_tokens: int | None = ...,
    ) -> Any: ...


def _field(raw: Any, name: str, default: str = "") -> str:
    """从 dict 或对象取字段，统一返回字符串。"""
    if isinstance(raw, dict):
        return str(raw.get(name, default) or default).strip()
    return str(getattr(raw, name, default) or default).strip()


def _recent_meta(message: Any) -> list[str]:
    """组装消息的元数据片段：时间戳、状态标签。"""
    meta: list[str] = []
    timestamp = getattr(message, "timestamp", None) if not isinstance(message, dict) else message.get("timestamp")
    if timestamp is not None:
        try:
            meta.append(f"time={timestamp.isoformat()}")
        except Exception:
            meta.append(f"time={timestamp}")
    tag = _field(message, "state_summary_tag", "none")
    if tag and tag != "none":
        meta.append(f"state_tag={tag}")
    return meta


def format_recent_entries(messages: Iterable[Any]) -> str:
    """把近期消息列表序列化为 "1) ...\n---\n2) ..." 文本。

    空 content 的消息跳过；元数据（时间/状态）作为后缀。
    """
    lines: list[str] = []
    for index, message in enumerate(messages, 1):
        content = _field(message, "content")
        if not content:
            continue
        meta = _recent_meta(message)
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"[{index}]{suffix} {content}")
    return "\n---\n".join(lines)


def _build_prompt(
    new_message: str,
    recent_text: str,
    new_state_summary_tag: str,
) -> list[dict[str, str]]:
    """构造 LLM 调用的 messages 列表。"""
    return [
        {
            "role": "system",
            "content": (
                "你是主动推送去重判官。判断新消息是否与近期推送过雷同。"
                "仅返回严格 JSON：{\"is_duplicate\": bool, \"reason\": \"<简短说明>\"}。"
                "无第二字段。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"新消息状态标签：{new_state_summary_tag}\n"
                f"近期推送：\n{recent_text or '（无）'}\n\n"
                f"新消息：\n{new_message}"
            ),
        },
    ]


class MessageDeduper:
    """基于 LLM 的主动推送去重器。"""

    def __init__(
        self,
        *,
        provider: _ChatLike,
        model: str,
        max_tokens: int = 256,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    async def is_duplicate(
        self,
        new_message: str,
        recent_proactive: list[Any] | None = None,
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]:
        """判断 new_message 与 recent_proactive 是否雷同。

        Returns:
            (is_duplicate, reason)
        异常 / 解析失败时按"非重复"放行（false, str(exc)）。
        """
        recent = list(recent_proactive or [])
        if not recent:
            return False, "无近期主动消息，放行"
        recent_text = format_recent_entries(recent)
        try:
            response = await self._provider.chat(
                messages=_build_prompt(new_message, recent_text, new_state_summary_tag),
                tools=[],
                model=self._model,
                max_tokens=min(DUPE_MAX_TOKENS_CAP, self._max_tokens),
            )
            payload = extract_json_object((getattr(response, "content", "") or "").strip())
        except Exception as exc:
            logger.warning("[proactive.deduper] 检测失败，放行: %s", exc)
            return False, str(exc)

        is_dup = bool(payload.get("is_duplicate", False))
        reason = str(payload.get("reason", ""))
        logger.info(
            "[proactive.deduper] duplicate=%s reason=%s",
            is_dup,
            reason,
        )
        return is_dup, reason
