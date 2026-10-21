"""主动链路漂移工具集 —— 漂移运行中暴露给 LLM 的受限工具。

设计思想（与 a20b45af proactive_v2/drift_tools.py 对齐）：

  - DriftRunContext：一次漂移运行的轻量状态
      message_sent —— message_push 是否已成功（单次 run 限一次）
      finished     —— finish_drift 是否已调用（loop 据此立即结束）
  - PushMessageTool（message_push）：
      发消息给用户；channel/chat_id 由配置预设可省略；
      image/media 归一化为路径列表；空消息+空媒体拒绝；
      已发过一次再调 → 拒绝（单次 run 限额）
  - FinishDriftTool（finish_drift）：终止工具
      skill 名必须合法；one_line / next 必填；
      message_result=sent 要求先成功 message_push，
      =silent 则不允许已 push（防状态撒谎）；
      校验全过后写 drift 状态并置 finished
  - MountServerTool（mount_server）：
      把已连接 MCP server 的工具挂载进本次漂移可用集
  - BoundedWebFetchTool：包装既有 web_fetch 工具，
      把返回 JSON 的 text 截断到 max_chars 并标注 truncated

工具采用 arguments-dict 约定（async execute(arguments: dict)），
可直接注册进 ProactiveMind 的 ToolRegistry。

ProactiveMind 命名（initiative/proactive_drift_tools.py）独立于原仓库。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from initiative.proactive_drift_state import DriftStateStore

logger = logging.getLogger(__name__)

_VALID_MESSAGE_RESULTS = frozenset({"sent", "silent"})


@dataclass
class DriftRunContext:
    """一次漂移运行的轻量状态标志。"""

    now_utc: datetime = None  # type: ignore[assignment]
    message_sent: bool = False
    finished: bool = False
    mounted_servers: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.now_utc is None:
            self.now_utc = datetime.now(timezone.utc)
        if self.mounted_servers is None:
            self.mounted_servers = set()

    def reset(self) -> None:
        """重置运行标志（供测试 / 复用）。"""
        self.message_sent = False
        self.finished = False
        self.mounted_servers.clear()


def _ok(payload: dict[str, Any] | None = None) -> str:
    body = {"ok": True}
    if payload:
        body.update(payload)
    return json.dumps(body, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _normalize_media(
    *, image: str = "", media: list[str] | str | None = None
) -> list[str]:
    """image 单图 + media（字符串或列表）归一化为非空路径列表。"""
    paths: list[str] = []
    if image:
        paths.append(str(image).strip())
    if isinstance(media, str):
        paths.append(media.strip())
    elif media:
        paths.extend(str(item).strip() for item in media)
    return [p for p in paths if p]


class PushMessageTool:
    """message_push —— 向用户发一条消息（单次运行限一次）。"""

    def __init__(
        self,
        ctx: DriftRunContext,
        send_message_fn: Callable[[str, list[str]], Awaitable[bool]] | None,
    ) -> None:
        self._ctx = ctx
        self._send = send_message_fn

    @property
    def name(self) -> str:
        return "message_push"

    @property
    def description(self) -> str:
        return (
            "向用户发送一条消息，可附带图片。单次漂移运行最多只能调用一次。"
            "channel 和 chat_id 已由配置预设，可省略不填。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "要发送的消息内容"},
                "image": {
                    "type": "string",
                    "description": "要发送的一张图片本地路径或 URL",
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "随消息发送的图片路径或 URL 列表",
                },
            },
            "required": [],
        }

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any] | None = None) -> str:
        args = dict(arguments or {})
        text = str(args.get("message") or args.get("content") or "").strip()
        media_paths = _normalize_media(
            image=str(args.get("image") or ""),
            media=args.get("media"),
        )

        if self._send is None:
            logger.info("[drift_tools] message_push 未配置发送通道")
            return _err("message_push not configured")
        if self._ctx.message_sent:
            logger.info("[drift_tools] message_push 拒绝：本次运行已用过")
            return _err("message_push already used in this drift run")
        if not text and not media_paths:
            logger.info("[drift_tools] message_push 拒绝：消息与媒体均为空")
            return _err("message or media is required")

        try:
            ok = await self._send(text, media_paths)
        except Exception as exc:
            logger.warning("[drift_tools] message_push 异常: %s", exc)
            return _err("message_push failed")
        if not ok:
            logger.warning("[drift_tools] message_push 发送失败")
            return _err("message_push failed")

        self._ctx.message_sent = True
        logger.info("[drift_tools] message_push ok")
        return _ok()


class FinishDriftTool:
    """finish_drift —— 终止工具：保存进度并结束本次漂移。"""

    def __init__(self, ctx: DriftRunContext, store: DriftStateStore) -> None:
        self._ctx = ctx
        self._store = store

    @property
    def name(self) -> str:
        return "finish_drift"

    @property
    def description(self) -> str:
        return "终止工具：结束本次漂移并保存进度状态。调用后循环立即结束。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_used": {"type": "string"},
                "one_line": {"type": "string"},
                "next": {"type": "string"},
                "message_result": {
                    "type": "string",
                    "enum": ["sent", "silent"],
                    "description": (
                        "sent 表示本轮已成功调用 message_push；"
                        "silent 表示本轮确认不该打扰用户，静默结束。"
                    ),
                },
                "note": {"type": "string"},
            },
            "required": ["skill_used", "one_line", "next", "message_result"],
        }

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any] | None = None) -> str:
        args = dict(arguments or {})
        skill_name = str(args.get("skill_used") or "").strip()
        summary = str(args.get("one_line") or "").strip()
        next_action = str(args.get("next") or "").strip()
        message_result = str(args.get("message_result") or "").strip()
        note_raw = args.get("note")
        note = str(note_raw).strip() if note_raw is not None else None

        if skill_name not in self._store.valid_skill_names():
            logger.info(
                "[drift_tools] finish_drift 拒绝未知 skill=%s", skill_name
            )
            return _err(f"unknown skill: {skill_name}")
        if not summary:
            return _err("one_line is required")
        if not next_action:
            return _err("next is required")
        if message_result not in _VALID_MESSAGE_RESULTS:
            return _err("message_result must be one of: sent, silent")
        if message_result == "sent" and not self._ctx.message_sent:
            return _err(
                "message_result=sent requires successful message_push first"
            )
        if message_result == "silent" and self._ctx.message_sent:
            return _err(
                "message_result=silent conflicts with successful message_push"
            )

        self._store.save_finish(
            skill_used=skill_name,
            one_line=summary,
            next_action=next_action,
            message_result=message_result,
            note=note,
            now_utc=self._ctx.now_utc,
        )
        self._ctx.finished = True
        logger.info(
            "[drift_tools] finish_drift ok: skill=%s one_line=%s",
            skill_name,
            summary[:120],
        )
        return _ok()


class MountServerTool:
    """mount_server —— 挂载已连接的 MCP server 工具到本次漂移。"""

    def __init__(self, source_tools: Any, mounted: set[str]) -> None:
        self._source = source_tools  # 提供 list_server_tools(server) 的对象
        self._mounted = mounted

    @property
    def name(self) -> str:
        return "mount_server"

    @property
    def description(self) -> str:
        return (
            "挂载一个已连接的 MCP server，使其工具在本次漂移中可用。"
            "挂载后即可直接调用该 server 的工具。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "要挂载的 MCP server 名称",
                },
            },
            "required": ["server"],
        }

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any] | None = None) -> str:
        args = dict(arguments or {})
        server = str(args.get("server") or "").strip()
        if not server:
            return _err("server is required")

        names = self._source.list_server_tools(server)
        if not names:
            return _err(f"MCP server '{server}' 不存在或未连接")

        new = set(names) - self._mounted
        if not new:
            return _ok({
                "message": f"'{server}' 已挂载，无新增工具",
                "tools": sorted(names),
            })
        self._mounted |= new
        logger.info(
            "[drift_tools] mount_server ok: server=%s new=%s",
            server,
            sorted(new),
        )
        return json.dumps(
            {"ok": True, "tools": sorted(names), "new": sorted(new)},
            ensure_ascii=False,
        )


class BoundedWebFetchTool:
    """包装既有 web_fetch 工具，把返回 JSON 的 text 截断到 max_chars。"""

    def __init__(self, wrapped: Any, max_chars: int) -> None:
        self._wrapped = wrapped
        self._max_chars = max(1, int(max_chars))

    @property
    def name(self) -> str:
        return self._wrapped.name

    @property
    def description(self) -> str:
        return self._wrapped.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._wrapped.parameters

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any] | None = None) -> str:
        result = await self._wrapped.execute(arguments)
        if not isinstance(result, str):
            return result
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            return result
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or len(text) <= self._max_chars:
            return result
        payload["text"] = text[: self._max_chars]
        payload["length"] = len(payload["text"])
        payload["truncated"] = True
        payload["note"] = (
            f"内容已截断至 {self._max_chars} 字符，"
            "如需更多内容请缩小范围或改用更精确的读取方式"
        )
        return json.dumps(payload, ensure_ascii=False)
