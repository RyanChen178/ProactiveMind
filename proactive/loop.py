"""主动推送 —— ProactiveLoop 定时轮询骨架。

每轮 tick 流程：
  1. Gate    —— 检查是否应该执行（被动回复忙、冷却中、概率跳过）
  2. Decide  —— 判断是否有内容值得推送（MVP 阶段留空，后续接数据源）
  3. Deliver —— 发送消息（MVP 阶段留空，后续接渠道）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, UTC

from proactive.energy import compute_urgency, next_interval
from proactive.presence import PresenceStore

log = logging.getLogger(__name__)


@dataclass
class TickResult:
    """一轮主动推送的结果。"""

    tick_id: str
    action: str  # "skipped" | "executed" | "no_content"
    urgency: float
    interval_s: float
    reason: str = ""


class ProactiveLoop:
    """主动推送定时循环。"""

    def __init__(
        self,
        presence: PresenceStore,
        *,
        is_passive_busy=None,
        max_ticks: int | None = None,
    ) -> None:
        self._presence = presence
        self._is_passive_busy = is_passive_busy or (lambda: False)
        self._max_ticks = max_ticks
        self._running = False
        self._tick_count = 0

    async def run(self) -> None:
        """启动轮询循环。"""
        self._running = True
        log.info("ProactiveLoop 启动")

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

        log.info("ProactiveLoop 停止")

    async def _tick(self) -> TickResult:
        """执行一轮主动推送检查。"""
        tick_id = f"tick-{self._tick_count}"
        now = datetime.now(UTC)
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

        # Decide: MVP 阶段没有数据源，总是 no_content
        log.info(
            "%s 完成: urgency=%.2f 间隔=%.0fs 下次约 %.0fs 后",
            tick_id, urgency, interval, interval,
        )
        return TickResult(
            tick_id=tick_id, action="no_content",
            urgency=urgency, interval_s=interval,
        )

    def stop(self) -> None:
        self._running = False
