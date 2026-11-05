"""主动链路数据网关 —— 三路并行预取 + 正文按需缓存。

设计思想（与 a20b45af proactive_v2/gateway.py 对齐）：

  - 在决策开始前把全部输入并行预取完（tick 的静态快照）：
      alerts  —— 直接透传完整内容
      context —— 直接透传完整内容
      content —— 双层结构：
          content_meta   轻量元数据列表（id / title / source / url /
                         published_at），决策时只看这个
          content_store  "{ack_server}:{id}" 复合键 → 正文的缓存，
                         正文由 web 抓取工具并行预取（失败存空串），
                         决策中途按需读取
  - 并行策略：三路 gather + content 路内部每条 URL 再 gather
  - 容错策略：任何一路 / 任何一条 URL 失败都降级
    （空列表 / 空字符串），绝不拖垮其它数据
  - 正文统一截断到 max_chars

web_fetch_fn 注入约定：接受 url 位置参数、可等待、返回正文文本
（返回空值视为失败）。比绑定具体工具对象更通用。

ProactiveMind 命名（initiative/proactive_gateway.py）独立于原仓库。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 8000
DEFAULT_CONTENT_LIMIT = 5


@dataclass
class GatewayResult:
    """一轮 tick 的静态输入快照。"""

    alerts: list[dict] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)
    # 轻量元数据（id 为复合键），按拉取顺序
    content_meta: list[dict] = field(default_factory=list)
    # 复合键 → 正文（抓取失败为空字符串）
    content_store: dict[str, str] = field(default_factory=dict)


def _compound_key(event: dict[str, Any]) -> str:
    """事件 → "{ack_server}:{id}" 复合键。"""
    item_id = str(
        event.get("event_id") or event.get("id") or ""
    )
    ack_server = str(event.get("ack_server") or event.get("source") or "")
    return f"{ack_server}:{item_id}"


class DataGateway:
    """三路并行预取网关。"""

    def __init__(
        self,
        *,
        alert_fn: Callable[[], Awaitable[list[dict]]] | None = None,
        feed_fn: Callable[[int], Awaitable[list[dict]]] | None = None,
        context_fn: Callable[[], Awaitable[list[dict]]] | None = None,
        web_fetch_fn: Callable[[str], Awaitable[str]] | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        content_limit: int = DEFAULT_CONTENT_LIMIT,
    ) -> None:
        self._alert_fn = alert_fn
        self._feed_fn = feed_fn
        self._context_fn = context_fn
        self._web_fetch_fn = web_fetch_fn
        self._max_chars = max(1, int(max_chars))
        self._content_limit = max(0, int(content_limit))

    async def run(self) -> GatewayResult:
        """并行预取三路，返回静态快照；单源失败不影响其它源。"""
        alerts, ctx_data, (meta, store) = await asyncio.gather(
            self._fetch_alerts(),
            self._fetch_context(),
            self._fetch_content(),
        )
        logger.info(
            "[gateway] 完成: alerts=%d context=%d content=%d (含正文=%d)",
            len(alerts), len(ctx_data), len(meta),
            sum(1 for v in store.values() if v),
        )
        return GatewayResult(
            alerts=alerts, context=ctx_data,
            content_meta=meta, content_store=store,
        )

    async def _fetch_alerts(self) -> list[dict]:
        if self._alert_fn is None:
            return []
        try:
            return list(await self._alert_fn() or [])
        except Exception as exc:
            logger.warning("[gateway] alerts 拉取失败: %s", exc)
            return []

    async def _fetch_context(self) -> list[dict]:
        if self._context_fn is None:
            return []
        try:
            return list(await self._context_fn() or [])
        except Exception as exc:
            logger.warning("[gateway] context 拉取失败: %s", exc)
            return []

    async def _fetch_content(self) -> tuple[list[dict], dict[str, str]]:
        """拉内容事件 → 并行抓正文 → (meta 列表, 正文缓存)。"""
        if self._feed_fn is None:
            return [], {}
        try:
            events = list(
                await self._feed_fn(self._content_limit) or []
            )
        except Exception as exc:
            logger.warning("[gateway] content 拉取失败: %s", exc)
            return [], {}
        if not events:
            return [], {}

        # 每条 URL 并行预取正文（失败/无 URL → 空串占位）
        bodies = await asyncio.gather(
            *(self._fetch_one(event.get("url", "")) for event in events),
            return_exceptions=True,
        )

        meta: list[dict] = []
        store: dict[str, str] = {}
        for event, body in zip(events, bodies):
            key = _compound_key(event)
            meta.append({
                "id": key,
                "title": str(event.get("title") or ""),
                "source": str(event.get("source") or ""),
                "url": str(event.get("url") or ""),
                "published_at": str(event.get("published_at") or ""),
            })
            if isinstance(body, BaseException) or not body:
                store[key] = ""
            else:
                store[key] = str(body)[: self._max_chars]
        return meta, store

    async def _fetch_one(self, url: Any) -> str:
        """抓单个 URL 正文，截断到 max_chars；任何失败返回空串。"""
        text_url = str(url or "").strip()
        if not text_url or self._web_fetch_fn is None:
            return ""
        try:
            body = await self._web_fetch_fn(text_url)
        except Exception as exc:
            logger.debug("[gateway] 正文抓取失败 %s: %s", text_url, exc)
            return ""
        if not body:
            return ""
        return str(body)[: self._max_chars]
