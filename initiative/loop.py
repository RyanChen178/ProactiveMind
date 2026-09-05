"""主动推送 —— InitiativeLoop 定时轮询骨架。

每轮 tick 流程：
  1. Gate     —— 检查是否应该执行（被动回复忙、冷却中）
  2. Decide   —— 依次查询三路数据源（alert > content > context），
                 命中即推送；无内容则进入 Drift 空闲任务
  3. Deliver  —— 发送消息到 push_callback
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from initiative.data_sources import DataSourceItem, DataSourceManager
from initiative.drift import WanderLoop
from initiative.energy import compute_urgency, next_interval
from initiative.presence import PresenceStore

log = logging.getLogger(__name__)

PushCallback = Callable[[str], Awaitable[None]]


@dataclass
class TickResult:
    """一轮主动推送的结果。"""

    tick_id: str
    action: str  # "skipped" | "executed" | "no_content" | "drift" | "pushed"
    urgency: float
    interval_s: float
    reason: str = ""
    pushed_content: str = ""
    pushed_source: str = ""  # 数据源类型 "alert"/"content"/"context"/"drift"


class InitiativeLoop:
    """主动推送定时循环。"""

    def __init__(
        self,
        presence: PresenceStore,
        *,
        is_passive_busy=None,
        drift_loop: WanderLoop | None = None,
        data_source_manager: DataSourceManager | None = None,
        push_callback: PushCallback | None = None,
        max_ticks: int | None = None,
    ) -> None:
        self._presence = presence
        self._is_passive_busy = is_passive_busy or (lambda: False)
        self._drift_loop = drift_loop
        self._data_source_manager = data_source_manager
        self._push_callback = push_callback
        self._max_ticks = max_ticks
        self._running = False
        self._tick_count = 0

    async def run(self) -> None:
        """启动轮询循环。"""
        self._running = True
        log.info("InitiativeLoop 启动")

        while self._running:
            last_user_at = self._presence.get_last_user_at()
            if last_user_at is None:
                # 从未交互过，等久一点
                await asyncio.sleep(300)
                continue

            interval = next_interval(last_user_at)
            await asyncio.sleep(interval)
            if not self._running:
                break

            await self._tick()
            self._tick_count += 1
            if self._max_ticks and self._tick_count >= self._max_ticks:
                break

        log.info("InitiativeLoop 停止")

    async def _tick(self) -> TickResult:
        """执行一轮主动推送检查。"""
        tick_id = f"tick-{self._tick_count}"
        now = datetime.now(timezone.utc)
        last_user_at = self._presence.get_last_user_at()
        assert last_user_at is not None

        urgency = compute_urgency(last_user_at, now)
        interval = next_interval(last_user_at, now)

        # Gate: 被动回复忙时跳过
        if self._is_passive_busy():
            result = TickResult(
                tick_id=tick_id, action="skipped",
                urgency=urgency, interval_s=interval,
                reason="passive_busy",
            )
            log.info("%s 跳过: 被动回复进行中", tick_id)
            return result

        # Gate: 刚发过主动消息不久，跳过
        last_proactive = self._presence.get_last_proactive_at()
        if last_proactive is not None:
            elapsed = (now - last_proactive).total_seconds()
            if elapsed < interval:
                result = TickResult(
                    tick_id=tick_id, action="skipped",
                    urgency=urgency, interval_s=interval,
                    reason="cooldown",
                )
                log.info("%s 跳过: 冷却中（%.0fs/%.0fs）", tick_id, elapsed, interval)
                return result

        # Decide: 优先查询三路数据源（alert/content/context）
        if self._data_source_manager is not None:
            push_result = await self._try_push_from_data_source(tick_id, now)
            if push_result is not None:
                return push_result

        # 数据源无内容时回退到 Drift 空闲任务
        if self._drift_loop is not None:
            drift_result = await self._drift_loop.run()

            # Drift 执行了 skill 且有 push_callback → 推送到客户端
            if (
                drift_result.action == "executed"
                and drift_result.summary
                and self._push_callback is not None
            ):
                push_text = (
                    f"[Drift · {drift_result.skill_name}]\n{drift_result.summary}"
                )
                try:
                    await self._push_callback(push_text)
                    self._presence.record_proactive(now)
                    log.info(
                        "%s 推送: skill=%s", tick_id, drift_result.skill_name
                    )
                    return TickResult(
                        tick_id=tick_id, action="pushed",
                        urgency=urgency, interval_s=interval,
                        reason="drift_push",
                        pushed_content=push_text,
                        pushed_source="drift",
                    )
                except Exception as exc:
                    log.warning("%s 推送失败: %s", tick_id, exc)

            action = (
                "drift" if drift_result.action == "executed"
                else "no_content"
            )
            log.info(
                "%s Drift: %s skill=%s",
                tick_id, drift_result.action, drift_result.skill_name or "—",
            )
            return TickResult(
                tick_id=tick_id, action=action,
                urgency=urgency, interval_s=interval,
                reason=drift_result.action,
            )

        log.info(
            "%s 完成: urgency=%.2f 间隔=%.0fs 下次约 %.0fs 后",
            tick_id, urgency, interval, interval,
        )
        return TickResult(
            tick_id=tick_id, action="no_content",
            urgency=urgency, interval_s=interval,
        )

    async def _try_push_from_data_source(
        self, tick_id: str, now: datetime
    ) -> TickResult | None:
        """从三路数据源中按优先级尝试推送一条内容。

        Returns:
            已推送时返回 TickResult(action="pushed")；无内容时返回 None。
        """
        if self._data_source_manager is None or self._push_callback is None:
            return None

        try:
            items: list[DataSourceItem] = await self._data_source_manager.fetch_all()
        except Exception as exc:
            log.warning("%s 数据源 fetch 失败: %s", tick_id, exc)
            return None

        if not items:
            log.debug("%s 数据源为空", tick_id)
            return None

        # 取优先级最高的条目
        top = items[0]
        urgency = compute_urgency(
            self._presence.get_last_user_at() or now, now
        )
        interval = next_interval(
            self._presence.get_last_user_at() or now, now
        )

        # 格式化推送文本：标记数据源类型便于用户识别
        prefix = {
            "alert": "[Alert]",
            "content": "[Content]",
            "context": "[Context]",
        }.get(top.source, f"[{top.source}]")
        push_text = f"{prefix} {top.content}"

        try:
            await self._push_callback(push_text)
            self._presence.record_proactive(now)
            log.info(
                "%s 数据源推送: source=%s priority=%.2f",
                tick_id, top.source, top.priority,
            )
            return TickResult(
                tick_id=tick_id,
                action="pushed",
                urgency=urgency,
                interval_s=interval,
                reason=f"data_source:{top.source}",
                pushed_content=push_text,
                pushed_source=top.source,
            )
        except Exception as exc:
            log.warning("%s 数据源推送失败: %s", tick_id, exc)
            return None

    def stop(self) -> None:
        self._running = False
