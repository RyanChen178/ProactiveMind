"""主动推送 —— 电量模型。

根据用户最后活跃时间计算"互动余温"：
  越久没说话，能量越低，饥渴度越高，轮询间隔越短。
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone


def compute_energy(
    last_user_at: datetime | None, now: datetime | None = None
) -> float:
    """计算互动余温能量值。

    E(t) = 0.50·exp(-t/30min) + 0.35·exp(-t/240min) + 0.15·exp(-t/2880min)

    三段衰减分别对应短时余温、中时语境、长时关系连续性(48h)。
    从未收到消息（None）时返回 0.0。
    """

    if last_user_at is None:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    # 统一处理 aware/naive datetime
    if last_user_at.tzinfo is None:
        last_user_at = last_user_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

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


def d_energy(energy: float) -> float:
    """互动饥渴度：energy 越低 → 饥渴度越高。

    线性映射：D = 1 - energy，越界裁到 [0, 1]。
    """
    if energy < 0.0:
        return 1.0
    if energy > 1.0:
        return 0.0
    return 1.0 - energy


def d_recent(msg_count: int, scale: float = 10.0) -> float:
    """对话语境丰富度：近期消息越多 → 越丰富（接近 1）。

    对数归一化：D = log(1 + k) / log(1 + scale)，上限 1。
    scale=10 时参考：0→0.00, 5→0.59, 10→0.76, 20→0.92。
    """
    if msg_count <= 0 or scale <= 1.0:
        return 0.0
    return min(1.0, math.log1p(msg_count) / math.log1p(scale))


def next_tick_from_score(
    base_score: float,
    *,
    tick_s1: int = 2400,
    tick_s0: int = 4800,
    tick_jitter: float = 0.3,
    threshold: float = 0.20,
) -> int:
    """根据 base_score 返回下一次 tick 的等待秒数（含随机抖动）。

    base_score > threshold → 短间隔 (tick_s1)
    base_score ≤ threshold → 长间隔 (tick_s0)
    base_score 越高 → 间隔越短 → 单位时间内抽签次数越多 → 越快触发。
    """
    base = tick_s1 if base_score > threshold else tick_s0
    if tick_jitter <= 0:
        return base
    offset = base * tick_jitter * (random.random() * 2 - 1)
    return max(1, int(base + offset))
