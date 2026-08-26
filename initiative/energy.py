"""主动推送 —— 电量模型。

根据用户最后活跃时间计算"互动余温"：
  越久没说话，能量越低，饥渴度越高，轮询间隔越短。
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, UTC


def compute_energy(
    last_user_at: datetime, now: datetime | None = None
) -> float:
    """计算互动余温能量值。

    E(t) = 0.50·exp(-t/30min) + 0.35·exp(-t/240min) + 0.15·exp(-t/2880min)

    三段衰减分别对应短时余温、中时语境、长时关系连续性(48h)。
    """

    if now is None:
        now = datetime.now(UTC)

    # 统一处理 aware/naive datetime
    if last_user_at.tzinfo is None:
        last_user_at = last_user_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    elapsed = (now - last_user_at).total_seconds()
    if elapsed < 0:
        return 1.0

    t = max(elapsed, 1.0)
    return (
        0.50 * math.exp(-t / 1800)      # 30 min
        + 0.35 * math.exp(-t / 14400)   # 240 min
        + 0.15 * math.exp(-t / 172800)  # 2880 min (48h)
    )


def compute_urgency(last_user_at: datetime, now: datetime | None = None) -> float:
    """计算主动推送饥渴度：D = 1 - energy。"""

    return 1.0 - compute_energy(last_user_at, now)


def next_interval(
    last_user_at: datetime,
    now: datetime | None = None,
    *,
    base_threshold: float = 0.20,
    interval_idle_s: int = 4800,
    interval_active_s: int = 2400,
    jitter: float = 0.3,
) -> float:
    """根据饥渴度决定下次轮询间隔（秒）。

    饥渴度高于阈值 → 间隔更短（interval_active_s）
    饥渴度低于阈值 → 间隔更长（interval_idle_s）
    加 ±jitter 随机抖动避免节拍过于规律。
    """

    urgency = compute_urgency(last_user_at, now)
    base = interval_active_s if urgency > base_threshold else interval_idle_s
    offset = base * jitter * (random.random() * 2 - 1)
    return max(base + offset, 60.0)
