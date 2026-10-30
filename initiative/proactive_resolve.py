"""主动链路消解模块 —— 决策收尾：三段查重后落成 send/skip。

设计思想（与 a20b45af proactive_v2/modules_resolve.py 对齐）：

  - 三段决策（顺序短路）：
      1. 终止动作非 reply —— 直接 skip（透传 skip_reason）
      2. 投递级重复 —— 按 delivery_key（引用指纹）在
         delivery_dedupe_hours 窗口内查重，重复则 skip
      3. 消息级重复 —— MessageDeduper 与近期推送做雷同判断，
         重复则 skip（可由 message_dedupe_enabled 关闭）
      全过 → send（附 delivery_key 供投递落库）
  - delivery_key 指纹：被引用条目的 URL（归一化：小写 host、
      去尾斜杠、丢 fragment）优先，其次 source|title，最后退回
      cited ids 排序；无任何引用时用最终消息前 500 字符。
      sha1 取前 16 位
  - ack TTL 分级（确认数据源的时长）：
      被引用 content 168h / 未引用但感兴趣 24h /
      明确丢弃 720h / 兜底守卫失败 24h
  - ProactiveResolveModule：phase 壳（slot=resolve /
      phase=proactive.resolve），读 judge 写入的终态，
      把 decision + draft_message 写回 frame.slots 供 deliver

复用 initiative.proactive_deduper.MessageDeduper。

ProactiveMind 命名（initiative/proactive_resolve.py）独立于原仓库。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from initiative.proactive_frame import ProactiveFrame

logger = logging.getLogger(__name__)

CITED_ACK_TTL_HOURS = 168
UNCITED_ACK_TTL_HOURS = 24
DISCARDED_ACK_TTL_HOURS = 720
POST_GUARD_ACK_TTL_HOURS = 24

ACTION_SEND = "send"
ACTION_SKIP = "skip"


@dataclass
class ResolveDecision:
    """消解结果。"""

    action: str  # "send" | "skip"
    reason: str = ""  # no_content / delivery_dedupe / message_dedupe / passed
    delivery_key: str = ""
    message: str = ""


def normalize_url(raw: str) -> str:
    """URL 归一化：小写 scheme/host、去尾斜杠、丢 fragment 与多余部分。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    path = parts.path.rstrip("/") or parts.path
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )


def build_delivery_key(
    *,
    cited_ids: list[str],
    content_meta: list[dict[str, Any]],
    final_message: str,
) -> str:
    """构造投递指纹：引用 URL > source|title > cited ids > 消息前缀。"""
    meta_map = {
        str(item.get("id") or ""): item for item in (content_meta or [])
    }
    refs: list[str] = []
    for key in sorted(set(cited_ids or [])):
        meta = meta_map.get(key)
        if meta is None:
            refs.append(f"id:{key}")
            continue
        url = normalize_url(str(meta.get("url") or ""))
        if url:
            refs.append(f"url:{url}")
            continue
        source = str(meta.get("source") or "").strip().lower()
        title = str(meta.get("title") or "").strip().lower()
        if title:
            refs.append(f"title:{source}|{title}")
            continue
        refs.append(f"id:{key}")

    refs = sorted(set(refs))
    if refs and any(not r.startswith("id:") for r in refs):
        key_src = json.dumps(refs)
    elif cited_ids:
        key_src = json.dumps(sorted(set(cited_ids)))
    else:
        key_src = str(final_message or "")[:500]
    return sha1(key_src.encode("utf-8")).hexdigest()[:16]


class ProactiveResolver:
    """三段查重决策器。"""

    def __init__(
        self,
        *,
        deduper: Any = None,
        message_dedupe_enabled: bool = True,
        delivery_dedupe_hours: int = 24,
        is_delivery_duplicate_fn: Callable[[str, int], bool] | None = None,
        recent_proactive_fn: Callable[[], list] = lambda: [],
    ) -> None:
        self._deduper = deduper
        self._message_dedupe_enabled = message_dedupe_enabled
        self._delivery_dedupe_hours = max(0, int(delivery_dedupe_hours))
        self._is_dup_fn = is_delivery_duplicate_fn
        self._recent_proactive_fn = recent_proactive_fn

    async def resolve(
        self,
        *,
        terminal_action: str,
        final_message: str,
        cited_ids: list[str] | None = None,
        content_meta: list[dict[str, Any]] | None = None,
        skip_reason: str = "",
    ) -> ResolveDecision:
        """按三段决策产出 send/skip。"""
        cited = list(cited_ids or [])
        if terminal_action != "reply":
            return ResolveDecision(
                action=ACTION_SKIP,
                reason=skip_reason or "no_content",
            )

        delivery_key = build_delivery_key(
            cited_ids=cited,
            content_meta=content_meta or [],
            final_message=final_message,
        )

        # 投递级重复（同指纹近期已发过）
        if self._is_dup_fn is not None and self._delivery_dedupe_hours > 0:
            try:
                if self._is_dup_fn(delivery_key, self._delivery_dedupe_hours):
                    return ResolveDecision(
                        action=ACTION_SKIP,
                        reason="delivery_dedupe",
                        delivery_key=delivery_key,
                        message=final_message,
                    )
            except Exception as exc:
                logger.warning("[resolve] 投递查重异常，放行: %s", exc)

        # 消息级重复（与近期推送语义雷同）
        if self._message_dedupe_enabled and self._deduper is not None:
            try:
                is_dup, reason = await self._deduper.is_duplicate(
                    final_message,
                    recent_proactive=self._recent_proactive_fn(),
                )
            except Exception as exc:
                logger.warning("[resolve] 消息查重异常，放行: %s", exc)
                is_dup, reason = False, str(exc)
            if is_dup:
                return ResolveDecision(
                    action=ACTION_SKIP,
                    reason="message_dedupe",
                    delivery_key=delivery_key,
                    message=final_message,
                )

        return ResolveDecision(
            action=ACTION_SEND,
            reason="passed",
            delivery_key=delivery_key,
            message=final_message,
        )


def ack_ttl_for(
    item_id: str,
    *,
    cited_ids: set[str],
    interesting_ids: set[str],
    discarded_ids: set[str],
) -> int | None:
    """按条目归属返回 ack TTL（小时）；不属于任何集合返回 None。"""
    if item_id in cited_ids:
        return CITED_ACK_TTL_HOURS
    if item_id in discarded_ids:
        return DISCARDED_ACK_TTL_HOURS
    if item_id in interesting_ids:
        return UNCITED_ACK_TTL_HOURS
    return None


def build_ack_plan(
    *,
    cited_ids: set[str],
    interesting_ids: set[str],
    discarded_ids: set[str],
) -> list[tuple[str, int]]:
    """生成 (item_id, ttl) 确认计划（排序稳定，供测试与审计）。"""
    plan: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item_id in sorted(
        cited_ids | interesting_ids | discarded_ids
    ):
        if item_id in seen:
            continue
        seen.add(item_id)
        ttl = ack_ttl_for(
            item_id,
            cited_ids=cited_ids,
            interesting_ids=interesting_ids,
            discarded_ids=discarded_ids,
        )
        if ttl is not None:
            plan.append((item_id, ttl))
    return plan


class ProactiveResolveModule:
    """resolve 阶段的 phase 壳。"""

    slot = "resolve"
    phase = "proactive.resolve"

    def __init__(self, resolver: ProactiveResolver) -> None:
        self._resolver = resolver

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        decision = await self._resolver.resolve(
            terminal_action=str(
                frame.slots.get("terminal_action") or ""
            ),
            final_message=str(frame.slots.get("final_message") or ""),
            cited_ids=frame.slots.get("cited_item_ids") or [],
            content_meta=frame.slots.get("proactive:content_meta") or [],
            skip_reason=str(frame.slots.get("skip_reason") or ""),
        )
        frame.slots["decision"] = decision.action
        frame.slots["draft_message"] = decision.message
        frame.slots["decision_reason"] = decision.reason
        frame.slots["delivery_key"] = decision.delivery_key
        return frame
