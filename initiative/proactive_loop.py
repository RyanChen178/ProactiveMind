"""主动链路主循环 —— 顶层装配：sleep → tick → 按分数排下一轮。

设计思想（与 a20b45af proactive_v2/loop.py 对齐）：

  - run() 循环结构：
      while running:
          interval = scheduler.next_interval(last_base_score)
          await sleep(interval)
          last_base_score = await tick()   # 异常捕获不终止循环
  - 单轮 tick = 构造新 frame → ProactivePhaseRunner.run(frame)
    （九阶段管线一次串通）→ 从 frame.slots 取 judge 写入的
    base_score 供下一轮排程（分数高 → 间隔短 → 更快触发）
  - tick 异常：记日志、分数置 None、继续循环（单轮故障
    不拖垮整个主动链路）
  - stop()：置 running=False，下一轮醒来即退出
  - sleep_fn / interval_fn / now_fn 全部可注入，
    测试用假时钟驱动不真睡
  - max_ticks：调试与测试用的轮次上限（生产不设）
  - tick_fn 可选注入：整体替换单轮执行体（如外部接管 LLM 环节），
    未注入时走默认的 runner 路径

复用 initiative.proactive_phases.ProactivePhaseRunner 与
initiative.proactive_schedule.ProactiveScheduler。

ProactiveMind 命名（initiative/proactive_loop.py）独立于原仓库。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from initiative.proactive_frame import ProactiveFrame, new_frame
from initiative.proactive_phases import ProactivePhaseRunner
from initiative.proactive_schedule import ProactiveScheduler, SLOT_BASE_SCORE

logger = logging.getLogger(__name__)

DEFAULT_FIRST_INTERVAL = 60


class ProactiveLoop:
    """主动链路定时 tick 主循环。"""

    def __init__(
        self,
        *,
        runner: ProactivePhaseRunner,
        scheduler: ProactiveScheduler,
        session_key: str = "",
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        tick_fn: Callable[[ProactiveFrame], Awaitable[ProactiveFrame]]
        | None = None,
        on_tick_error: Callable[[BaseException], None] | None = None,
        max_ticks: int | None = None,
        first_interval_s: float | None = None,
    ) -> None:
        self._runner = runner
        self._scheduler = scheduler
        self._session_key = session_key
        self._sleep = sleep_fn
        self._tick_fn = tick_fn
        self._on_tick_error = on_tick_error
        self._max_ticks = max_ticks
        self._first_interval = first_interval_s
        self._running = False
        self.tick_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """请求停止；下一轮醒来即退出。"""
        self._running = False

    def next_interval(self, base_score: float | None = None) -> int:
        """下一轮等待秒数（委托 scheduler）。"""
        return self._scheduler.next_interval(base_score)

    async def _tick(self) -> float | None:
        """执行单轮：构造 frame → 跑管线 → 取分数。"""
        frame = new_frame(self._session_key)
        if self._tick_fn is not None:
            frame = await self._tick_fn(frame)
        else:
            frame = await self._runner.run(frame)
        self.tick_count += 1
        score = frame.slots.get(SLOT_BASE_SCORE)
        return float(score) if isinstance(score, (int, float)) else None

    async def run(self) -> None:
        """主循环：sleep(interval) → tick，异常不终止。"""
        self._running = True
        self.tick_count = 0
        logger.info(
            "[proactive_loop] 启动 session=%s", self._session_key or "-"
        )
        last_score: float | None = None
        ticks = 0
        first = True
        try:
            while self._running:
                if first and self._first_interval is not None:
                    interval = self._first_interval
                else:
                    interval = self.next_interval(last_score)
                first = False
                logger.debug(
                    "[proactive_loop] 下轮 tick 间隔=%ss", interval
                )
                await self._sleep(max(0, interval))
                if not self._running:
                    break
                try:
                    last_score = await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("[proactive_loop] tick 异常")
                    if self._on_tick_error is not None:
                        try:
                            self._on_tick_error(exc)
                        except Exception:
                            pass
                    last_score = None
                ticks += 1
                if self._max_ticks is not None and ticks >= self._max_ticks:
                    break
        finally:
            self._running = False
            logger.info("[proactive_loop] 停止 ticks=%d", self.tick_count)
