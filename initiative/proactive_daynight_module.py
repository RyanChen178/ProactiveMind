"""昼夜 gate 模块 —— 夜间静默时段压低主动推送通过概率。

设计思想（与 a20b45af plugins/daynight_gate/plugin.py 对齐）：

  - 配置：本地时区 + 起止时间（HH:MM）+ 夜间通过概率
  - 白天（窗口外）：frame 原样透传，不写任何槽位
  - 夜间（窗口内，支持跨午夜如 22:00-06:00）：
      pass_probability = min(管线已有概率, 配置概率)
      —— 与其它 gate 模块叠加时取最严格值
      写三个槽位：
        proactive:gate:pass_probability  压低后的通过概率
        proactive:gate:reason            阻断原因（默认 quiet_hours）
        proactive:effect:daynight_gate   效应审计（时段/时区/概率）
  - 模块声明 slot=proactive.gate.daynight / phase=proactive.gate，
    可直接挂进 ProactivePhaseRunner

时间判定基于 frame.input.started_at 转本地时区，
不依赖系统时钟（可测）。

ProactiveMind 命名（initiative/proactive_daynight_module.py）
独立于原仓库，配置用 dataclass + 手写校验（不引 pydantic）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from initiative.proactive_frame import ProactiveFrame

SLOT_PASS_PROBABILITY = "proactive:gate:pass_probability"
SLOT_GATE_REASON = "proactive:gate:reason"
SLOT_EFFECT = "proactive:effect:daynight_gate"


def _parse_hhmm(value: str) -> time:
    """解析 HH:MM 为 time；格式错抛 ValueError。"""
    parts = str(value or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"时间格式必须是 HH:MM，得到: {value!r}")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"时间格式必须是 HH:MM，得到: {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"时间超出范围: {value!r}")
    return time(hour=hour, minute=minute)


def _in_window(now: time, start: time, end: time) -> bool:
    """判断当前时间是否在 [start, end) 窗口内（支持跨午夜）。"""
    if start == end:
        return False  # 零长度窗口
    if start < end:
        return start <= now < end
    # 跨午夜：22:00-06:00 → >=22:00 或 <06:00
    return now >= start or now < end


@dataclass(frozen=True)
class DayNightConfig:
    """昼夜 gate 配置。"""

    enabled: bool = True
    timezone: str = "Asia/Shanghai"
    start: str = "00:00"
    end: str = "06:00"
    pass_probability: float = 0.15
    reason: str = "quiet_hours"

    def __post_init__(self) -> None:
        if not (0.0 <= self.pass_probability <= 1.0):
            raise ValueError(
                f"pass_probability 必须在 [0, 1]，当前: {self.pass_probability}"
            )
        # 校验时间格式
        _parse_hhmm(self.start)
        _parse_hhmm(self.end)
        # 校验时区
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知时区: {self.timezone}") from exc


class DayNightGateModule:
    """夜间静默 gate：窗口内压低通过概率。"""

    slot = "proactive.gate.daynight"
    phase = "proactive.gate"

    def __init__(self, config: DayNightConfig | None = None) -> None:
        self._config = config or DayNightConfig()
        self._zone = ZoneInfo(self._config.timezone)
        self._start = _parse_hhmm(self._config.start)
        self._end = _parse_hhmm(self._config.end)

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        if not self._config.enabled:
            return frame

        local_now = frame.input.started_at.astimezone(self._zone)
        if not _in_window(local_now.time(), self._start, self._end):
            return frame  # 白天：不写任何槽位

        # 与其它 gate 叠加：取更严格的（更小）概率
        probability = self._config.pass_probability
        current = frame.slots.get(SLOT_PASS_PROBABILITY)
        if current is not None:
            probability = min(float(current), probability)

        frame.slots[SLOT_PASS_PROBABILITY] = probability
        frame.slots[SLOT_GATE_REASON] = self._config.reason
        frame.slots[SLOT_EFFECT] = {
            "provider_name": "daynight_gate",
            "pass_probability": probability,
            "reason": self._config.reason,
            "timezone": self._config.timezone,
            "start": self._config.start,
            "end": self._config.end,
        }
        return frame
