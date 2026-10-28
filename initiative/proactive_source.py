"""主动链路数据源模块 —— 三路取数 + 限额 + context 兜底门控。

设计思想（与 a20b45af proactive_v2/modules_source.py 对齐）：

  - source 阶段把外部数据源的三路内容预取进 frame.slots：
      alerts  —— 高优告警，直接透传（不限额、不筛选）
      content —— 内容流，截断到 content_limit（正文按需获取，
                 此处只放元数据列表）
      context —— 背景上下文，仅在 gate 开启兜底
                 （frame.slots 的 context_as_fallback_open）时携带
  - 每路独立容错：一路拉取异常降级为空列表，不阻断其它路
  - 数据源 duck typing：任何有 alert_source / content_source /
    context_source 三属性的对象（即已有的 DataSourceManager）
    或提供 fetch_alerts() / fetch_contents() / fetch_context()
    的对象都可注入
  - ack 回调：投递成功后按 compound key 确认数据源
    （"{source}:{id}" 格式），供上层在 deliver 后调用
  - ProactiveSourceModule：phase 壳（slot=source /
    phase=proactive.source），写 frame.slots 的三个约定 key

复用 initiative.data_sources.DataSourceManager 的三路语义。

ProactiveMind 命名（initiative/proactive_source.py）独立于原仓库。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from initiative.proactive_frame import ProactiveFrame

logger = logging.getLogger(__name__)

SLOT_ALERTS = "proactive:alerts"
SLOT_CONTENT_META = "proactive:content_meta"
SLOT_CONTEXT = "proactive:context"
SLOT_FALLBACK_OPEN = "context_as_fallback_open"

DEFAULT_CONTENT_LIMIT = 5


def _split_compound_key(compound_key: str) -> tuple[str, str] | None:
    """拆 "{source}:{id}" 复合键；格式不对返回 None。"""
    parts = str(compound_key or "").split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _to_plain(item: Any) -> dict[str, Any]:
    """把 DataSourceItem（dataclass）或 dict 统一转成普通 dict。"""
    from dataclasses import is_dataclass, asdict

    if isinstance(item, dict):
        return dict(item)
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    return dict(item)


class ProactiveSource:
    """三路数据源取数器（独立容错 + 限额 + context 门控）。"""

    def __init__(
        self,
        data_sources: Any,
        *,
        content_limit: int = DEFAULT_CONTENT_LIMIT,
    ) -> None:
        self._ds = data_sources
        self._content_limit = max(0, int(content_limit))

    async def fetch_alerts(self) -> list[dict[str, Any]]:
        """拉取告警路（直接透传，异常降级空列表）。"""
        try:
            source = getattr(self._ds, "alert_source", None)
            if source is None:
                return []
            items = await source.fetch()
            return [_to_plain(item) for item in (items or [])]
        except Exception as exc:
            logger.warning("[source] alerts 拉取失败: %s", exc)
            return []

    async def fetch_contents(self) -> list[dict[str, Any]]:
        """拉取内容路（截断到 content_limit，异常降级空列表）。"""
        try:
            source = getattr(self._ds, "content_source", None)
            if source is None:
                return []
            items = await source.fetch()
            return [_to_plain(item) for item in (items or [])][
                : self._content_limit
            ]
        except Exception as exc:
            logger.warning("[source] contents 拉取失败: %s", exc)
            return []

    async def fetch_context(self, *, fallback_open: bool) -> list[dict[str, Any]]:
        """拉取背景路；fallback 未开启时返回空（对齐兜底门控）。"""
        if not fallback_open:
            return []
        try:
            source = getattr(self._ds, "context_source", None)
            if source is None:
                return []
            items = await source.fetch()
            return [_to_plain(item) for item in (items or [])]
        except Exception as exc:
            logger.warning("[source] context 拉取失败: %s", exc)
            return []


class ProactiveSourceModule:
    """source 阶段的 phase 模块壳：三路预取写进 frame.slots。"""

    slot = "source"
    phase = "proactive.source"

    def __init__(
        self,
        source: ProactiveSource,
        *,
        ack_fn: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._source = source
        self._ack_fn = ack_fn

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        fallback_open = bool(frame.slots.get(SLOT_FALLBACK_OPEN, False))

        alerts = await self._source.fetch_alerts()
        contents = await self._source.fetch_contents()
        context = await self._source.fetch_context(
            fallback_open=fallback_open
        )

        frame.slots[SLOT_ALERTS] = alerts
        frame.slots[SLOT_CONTENT_META] = [
            self._to_meta(item) for item in contents
        ]
        frame.slots[SLOT_CONTEXT] = context
        return frame

    @staticmethod
    def _to_meta(item: dict[str, Any]) -> dict[str, Any]:
        """归一化 content 元数据（id / title / source / url）。"""
        return {
            "id": str(
                item.get("id")
                or item.get("item_id")
                or item.get("event_id")
                or ""
            ),
            "title": str(item.get("title") or item.get("content") or ""),
            "source": str(item.get("source") or ""),
            "url": str(item.get("url") or ""),
        }

    async def ack(self, compound_key: str) -> bool:
        """按复合键确认数据源；格式不对返回 False。"""
        if self._ack_fn is None:
            return False
        if _split_compound_key(compound_key) is None:
            logger.warning("[source] 无效复合键: %r", compound_key)
            return False
        try:
            await self._ack_fn(compound_key)
            return True
        except Exception as exc:
            logger.warning("[source] ack 失败 %s: %s", compound_key, exc)
            return False
