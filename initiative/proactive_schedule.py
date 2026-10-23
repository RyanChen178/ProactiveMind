"""主动链路调度模块 —— 计算下一轮 tick 的等待间隔。

设计思想（与 a20b45af proactive_v2/modules_schedule.py 对齐）：

  - 三档决策：
      1. 无 presence —— 固定间隔 cfg.interval_seconds（trace: fixed_no_presence）
      2. base_score 缺省 —— 从 presence 取 last_user_at 计算
         energy → d_energy × score_weight_energy 得到分数
      3. 有分数 —— next_tick_from_score（分数越高间隔越短，含抖动）
  - trace_fn 记录每次决策（base_score / interval / mode），
    供调试面板与策略追踪消费
  - 同时提供 phase 模块壳（slot="schedule" / phase="proactive.schedule"），
    可直接挂进 ProactivePhaseRunner：run(frame) 从 frame.slots 读
    judge 写入的 base_score，把算出的间隔写回 frame.slots

复用 initiative.energy 的 compute_energy / d_energy /
next_tick_from_score 与 proactive_config.ProactiveConfig（duck typing）。

ProactiveMind 命名（initiative/proactive_schedule.py）独立于原仓库。
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

from initiative.energy import compute_energy, d_energy, next_tick_from_score
from initiative.proactive_frame import ProactiveFrame

logger = logging.getLogger(__name__)

SLOT_NEXT_INTERVAL = "next_tick_interval"
SLOT_BASE_SCORE = "base_score"


class ProactiveScheduler:
    """下一轮 tick 间隔计算器。"""

    def __init__(
        self,
        *,
        cfg: Any,
        presence: Any = None,
        rng: random.Random | None = None,
        target_session_key_fn: Callable[[], str] = lambda: "",
        trace_fn: Callable[..., None] = lambda **kw: None,
    ) -> None:
        self._cfg = cfg
        self._presence = presence
        self._rng = rng if rng is not None else random.Random()
        self._target_session_key_fn = target_session_key_fn
        self._trace_fn = trace_fn

    def next_interval(self, base_score: float | None = None) -> int:
        """计算下一轮 tick 的等待秒数。

        Args:
            base_score: 决策分数（通常由 judge 写入）。
                None 时从 presence 推导（energy 贡献 × 权重）。

        Returns:
            等待秒数（>= 1）。
        """
        if self._presence is None:
            interval = int(getattr(self._cfg, "interval_seconds", 1800) or 1800)
            self._trace_fn(
                base_score=base_score,
                interval=interval,
                mode="fixed_no_presence",
            )
            return interval

        if base_score is None:
            session_key = self._target_session_key_fn()
            last_user_at = self._presence.get_last_user_at(session_key)
            energy = compute_energy(last_user_at)
            weight = float(
                getattr(self._cfg, "score_weight_energy", 0.40)
            )
            base_score = d_energy(energy) * weight

        interval = next_tick_from_score(
            base_score,
            tick_s1=int(getattr(self._cfg, "tick_interval_s1", 2400) or 2400),
            tick_s0=int(getattr(self._cfg, "tick_interval_s0", 4800) or 4800),
            tick_jitter=float(getattr(self._cfg, "tick_jitter", 0.3) or 0.0),
        )
        self._trace_fn(
            base_score=base_score,
            interval=interval,
            mode="adaptive",
        )
        return interval


class ProactiveScheduleModule:
    """schedule 阶段的 phase 模块壳：frame.slots 进出。"""

    slot = "schedule"
    phase = "proactive.schedule"

    def __init__(self, scheduler: ProactiveScheduler) -> None:
        self._scheduler = scheduler

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        """从 frame.slots 读 base_score（judge 写入），写回下一轮间隔。

        base_score 不存在时由调度器从 presence 自行推导。
        """
        base_score = frame.slots.get(SLOT_BASE_SCORE)
        interval = self._scheduler.next_interval(base_score)
        frame.slots[SLOT_NEXT_INTERVAL] = interval
        return frame
