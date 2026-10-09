"""主动链路 Sensor —— 感知层：从会话与 presence 收集近期上下文。

设计思想（与 a20b45af proactive_v2/sensor.py 对齐）：

  - target_session_key()：从 cfg 解析 channel/chat_id → 组合 session key
  - last_user_at()：委托 presence.Store.get_last_user_at
  - collect_recent()：取最近 N 条 user/assistant 消息（裁剪 200 字符）
  - collect_recent_proactive()：取最近 N 条 proactive=True 的 assistant 消息，
    按时间倒序
  - 异常安全：session 取不到 / message 字段缺失都不抛错，降级返回 []

ProactiveMind 命名（initiative/proactive_sensor.py）独立于原仓库，
不依赖 agent.prompting / session.manager 等内部模块，用 duck typing
接受任何具有 .get_or_create(key)→session 的对象 + .messages 列表的 session。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


CONTENT_PREVIEW_CHARS = 200


@dataclass
class RecentProactiveMessage:
    """一条近期主动推送消息的快照。"""

    content: str
    timestamp: datetime | None = None
    state_summary_tag: str = "none"
    source_refs: list[Any] = field(default_factory=list)


class _SessionLike(Protocol):
    """最小 session 协议：拥有 .messages 列表。"""

    messages: list[dict[str, Any]]


class _SessionsAPI(Protocol):
    def get_or_create(self, key: str) -> _SessionLike: ...


class _PresenceAPI(Protocol):
    def get_last_user_at(self, key: str) -> datetime | None: ...


@dataclass
class _SensorConfig:
    """Sensor 所需配置的最小子集（duck typing 接受任意含这些字段的对象）。"""

    default_channel: str = ""
    default_chat_id: str = ""
    recent_chat_messages: int = 20


class ProactiveSensor:
    """主动链路感知层：把会话与 presence 上下文打包给后续模块。"""

    def __init__(
        self,
        cfg: Any,
        sessions: _SessionsAPI,
        presence: _PresenceAPI | None = None,
    ) -> None:
        self._cfg = cfg
        self._sessions = sessions
        self._presence = presence

    def target_session_key(self) -> str:
        """从 cfg 解析 default_channel/default_chat_id → 组合 session key。

        缺一返回空串（表示无目标）。
        """
        channel = (getattr(self._cfg, "default_channel", "") or "").strip()
        chat_id = (getattr(self._cfg, "default_chat_id", "") or "").strip()
        if not channel or not chat_id:
            return ""
        return f"{channel}:{chat_id}"

    def last_user_at(self) -> datetime | None:
        """最后一次用户消息时间（来自 presence）。"""
        if self._presence is None:
            return None
        key = self.target_session_key()
        if not key:
            return None
        return self._presence.get_last_user_at(key)

    def collect_recent(self) -> list[dict]:
        """取最近 N 条 user/assistant 消息（裁剪 200 字符）。"""
        key = self.target_session_key()
        if not key:
            return []
        try:
            session = self._sessions.get_or_create(key)
        except Exception:
            return []
        limit = max(1, int(getattr(self._cfg, "recent_chat_messages", 20) or 20))
        messages = list(session.messages)[-limit:]
        results: list[dict] = []
        for message in messages:
            if message.get("role") not in ("user", "assistant"):
                continue
            content = str(message.get("content", "") or "")
            if not content or not content.strip():
                continue
            results.append({
                "role": message["role"],
                "content": content[:CONTENT_PREVIEW_CHARS],
                "timestamp": str(message.get("timestamp", "") or ""),
            })
        return results

    def collect_recent_proactive(self, n: int = 5) -> list[RecentProactiveMessage]:
        """取最近 N 条 proactive=True 的 assistant 消息（按时间倒序）。"""
        key = self.target_session_key()
        if not key:
            return []
        try:
            session = self._sessions.get_or_create(key)
        except Exception:
            return []
        results: list[RecentProactiveMessage] = []
        for message in reversed(session.messages):
            if message.get("role") != "assistant":
                continue
            if not message.get("proactive") or not message.get("content"):
                continue
            results.append(RecentProactiveMessage(
                content=str(message["content"]),
                timestamp=self._parse_timestamp(message.get("timestamp")),
                state_summary_tag=str(
                    message.get("state_summary_tag", "none") or "none"
                ),
                source_refs=list(message.get("source_refs") or []),
            ))
            if len(results) >= n:
                break
        return list(reversed(results))

    @staticmethod
    def _parse_timestamp(raw: Any) -> datetime | None:
        """解析 ISO 时间字符串；无 tz 时按本地时区补。失败返回 None。"""
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            ts = datetime.fromisoformat(text)
        except (ValueError, TypeError):
            return None
        if ts.tzinfo is None:
            return ts.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return ts
