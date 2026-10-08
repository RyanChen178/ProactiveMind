"""主动链路预设配置 —— 多种调参档位（daily / quiet / ...）。

设计思想（与 a20b45af proactive_v2/presets.py 对齐）：

  - 用 TypedDict 声明每段配置的字段结构（不引入 Pydantic，保持纯 stdlib）
  - 顶层 PRESETS 是 name → PresetConfig 的字典
  - ALLOWED_OVERRIDE_KEYS 白名单：哪些 key 允许用户/环境变量覆盖
  - STRATEGY_PARAMS：内部固定参数，不对外暴露
  - resolve_preset(name, overrides) 把白名单覆盖合并到基础预设上

ProactiveMind 命名（initiative/proactive_presets.py）独立于原仓库。
"""

from __future__ import annotations

from typing import TypedDict


class TriggerPreset(TypedDict):
    tick_interval_s0: int
    tick_interval_s1: int
    tick_jitter: float


class GatePreset(TypedDict):
    judge_send_threshold: float


class AnyActionPreset(TypedDict):
    anyaction_enabled: bool
    anyaction_daily_max_actions: int
    anyaction_min_interval_seconds: int
    anyaction_probability_min: float
    anyaction_probability_max: float
    anyaction_idle_scale_minutes: float
    anyaction_reset_hour_local: int
    anyaction_timezone: str


class SafetyPreset(TypedDict):
    delivery_dedupe_hours: int
    message_dedupe_recent_n: int


class ContextPreset(TypedDict):
    context_only_daily_max: int
    context_only_min_interval_hours: int


class PresetConfig(TypedDict):
    trigger: TriggerPreset
    gate: GatePreset
    anyaction: AnyActionPreset
    safety: SafetyPreset
    context: ContextPreset


# 预设档位定义
PRESETS: dict[str, PresetConfig] = {
    "daily": {
        "trigger": {
            "tick_interval_s0": 900,
            "tick_interval_s1": 300,
            "tick_jitter": 0.2,
        },
        "gate": {
            "judge_send_threshold": 0.60,
        },
        "anyaction": {
            "anyaction_enabled": True,
            "anyaction_daily_max_actions": 999,
            "anyaction_min_interval_seconds": 20,
            "anyaction_probability_min": 0.75,
            "anyaction_probability_max": 0.98,
            "anyaction_idle_scale_minutes": 15.0,
            "anyaction_reset_hour_local": 12,
            "anyaction_timezone": "Asia/Shanghai",
        },
        "safety": {
            "delivery_dedupe_hours": 1,
            "message_dedupe_recent_n": 5,
        },
        "context": {
            "context_only_daily_max": 20,
            "context_only_min_interval_hours": 1,
        },
    },
    "quiet": {
        "trigger": {
            "tick_interval_s0": 1800,
            "tick_interval_s1": 900,
            "tick_jitter": 0.3,
        },
        "gate": {
            "judge_send_threshold": 0.75,
        },
        "anyaction": {
            "anyaction_enabled": True,
            "anyaction_daily_max_actions": 12,
            "anyaction_min_interval_seconds": 600,
            "anyaction_probability_min": 0.05,
            "anyaction_probability_max": 0.30,
            "anyaction_idle_scale_minutes": 120.0,
            "anyaction_reset_hour_local": 12,
            "anyaction_timezone": "Asia/Shanghai",
        },
        "safety": {
            "delivery_dedupe_hours": 24,
            "message_dedupe_recent_n": 8,
        },
        "context": {
            "context_only_daily_max": 1,
            "context_only_min_interval_hours": 24,
        },
    },
}


# 内部固定参数（不对外暴露）
STRATEGY_PARAMS = {
    "score_weight_energy": 0.35,
    "message_dedupe_enabled": True,
    "recent_chat_messages": 20,
    "interval_seconds": 1800,
}


# 允许通过 overrides 覆盖的 key 白名单
ALLOWED_OVERRIDE_KEYS: dict[str, frozenset[str]] = {
    "trigger": frozenset({
        "tick_interval_s0", "tick_interval_s1", "tick_jitter",
    }),
    "gate": frozenset({
        "judge_send_threshold",
    }),
    "anyaction": frozenset({
        "anyaction_enabled", "anyaction_daily_max_actions",
        "anyaction_min_interval_seconds",
        "anyaction_probability_min", "anyaction_probability_max",
        "anyaction_idle_scale_minutes",
        "anyaction_reset_hour_local", "anyaction_timezone",
    }),
    "safety": frozenset({
        "delivery_dedupe_hours", "message_dedupe_recent_n",
    }),
    "context": frozenset({
        "context_only_daily_max", "context_only_min_interval_hours",
    }),
}


def get_preset(name: str) -> PresetConfig:
    """按名取预设；未找到时降级为 daily。"""
    if name in PRESETS:
        return PRESETS[name]
    return PRESETS["daily"]


def resolve_preset(
    name: str, overrides: dict[str, dict[str, object]] | None = None
) -> PresetConfig:
    """合并预设与白名单 overrides（非法 key 静默丢弃）。"""
    base = {section: dict(values) for section, values in get_preset(name).items()}
    if not overrides:
        return base  # type: ignore[return-value]

    for section, values in overrides.items():
        allowed = ALLOWED_OVERRIDE_KEYS.get(section, frozenset())
        if section not in base:
            continue
        for key, value in values.items():
            if key in allowed:
                base[section][key] = value
    return base  # type: ignore[return-value]


def preset_names() -> list[str]:
    """返回所有可用预设名（按声明顺序）。"""
    return list(PRESETS.keys())
