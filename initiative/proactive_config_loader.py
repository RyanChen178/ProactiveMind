"""主动链路配置加载与校验。

设计思想（与 a20b45af proactive_v2/config_loader.py 对齐）：

  - load_proactive_config(p) 从 dict 加载为 ProactiveConfig
  - 校验链（顺序执行）：
      1. 根级键白名单（禁止平铺旧键直接放根下）
      2. target 子块（channel / chat_id）类型校验
      3. profile 必填——引用 presets 中的预设名
      4. profiles 用户自定义：deep merge 到内置 PRESETS
      5. overrides 白名单校验（类别 + 键两层）
      6. 范围校验：阈值 [0,1] / pmin <= pmax / tick 间隔递减且 >= 1
      7. agent / drift 子块键白名单
  - 校验失败抛 ProactiveConfigError（带可读信息）
  - 最后把 preset + overrides + strategy params 展平成 ProactiveConfig

复用 initiative.proactive_presets 的 PRESETS / ALLOWED_OVERRIDE_KEYS /
STRATEGY_PARAMS。

ProactiveMind 命名（initiative/proactive_config_loader.py）独立于原仓库。
"""

from __future__ import annotations

import copy
from typing import Any

from initiative.proactive_config import ProactiveConfig
from initiative.proactive_presets import (
    ALLOWED_OVERRIDE_KEYS,
    PRESETS,
    STRATEGY_PARAMS,
)


class ProactiveConfigError(Exception):
    """主动链路配置错误。"""

    pass


# 允许出现在 proactive 根下的键（防止旧平铺键混入）
_ALLOWED_ROOT_KEYS = frozenset({
    "enabled",
    "profile",
    "profiles",
    "target",
    "model",
    "overrides",
    "agent",
    "drift",
    "feed",
    "feed_poller_interval_seconds",
})

# agent 子块允许的键（映射到 agent_tick_* 字段）
_ALLOWED_AGENT_KEYS = frozenset({
    "model",
    "max_steps",
    "content_limit",
    "web_fetch_max_chars",
    "context_prob",
    "delivery_cooldown_hours",
})

# drift 子块允许的键
_ALLOWED_DRIFT_KEYS = frozenset({
    "enabled",
    "max_steps",
    "min_interval_hours",
})


def _check_forbidden_root_keys(p: dict[str, Any]) -> None:
    forbidden = set(p.keys()) - _ALLOWED_ROOT_KEYS
    if forbidden:
        raise ProactiveConfigError(
            f"proactive 配置中出现非法的根级键: {', '.join(sorted(forbidden))}。"
            "请使用 profile / profiles / target 等分块方式配置。"
        )


def _validate_overrides(overrides: dict[str, Any]) -> None:
    """overrides 两层白名单校验：类别 + 键。"""
    for category, values in overrides.items():
        if category not in ALLOWED_OVERRIDE_KEYS:
            raise ProactiveConfigError(
                f"overrides 中的非法类别: '{category}'。"
                f"只允许: {', '.join(sorted(ALLOWED_OVERRIDE_KEYS))}"
            )
        if not isinstance(values, dict):
            raise ProactiveConfigError(
                f"overrides.{category} 必须是字典，"
                f"当前类型: {type(values).__name__}"
            )
        allowed = ALLOWED_OVERRIDE_KEYS[category]
        for key in values.keys():
            if key not in allowed:
                raise ProactiveConfigError(
                    f"overrides.{category}.{key} 不在白名单中。"
                    f"允许的键: {', '.join(sorted(allowed))}"
                )


def _validate_ranges(flat: dict[str, Any]) -> None:
    """数值范围校验：阈值 / 概率序 / tick 间隔递减。"""
    # 阈值类必须 [0, 1]
    for key in (
        "judge_send_threshold",
        "anyaction_probability_min",
        "anyaction_probability_max",
    ):
        if key in flat:
            val = flat[key]
            if not (0 <= val <= 1):
                raise ProactiveConfigError(
                    f"{key} 必须在 [0, 1] 范围内，当前值: {val}"
                )

    # pmin <= pmax
    pmin = flat.get("anyaction_probability_min")
    pmax = flat.get("anyaction_probability_max")
    if pmin is not None and pmax is not None and pmin > pmax:
        raise ProactiveConfigError(
            f"anyaction_probability_min ({pmin}) 不能大于 "
            f"anyaction_probability_max ({pmax})"
        )

    # tick_interval_s0 >= s1 >= 1（s0 是低分长间隔，s1 是高分短间隔）
    s0 = flat.get("tick_interval_s0")
    s1 = flat.get("tick_interval_s1")
    if s0 is not None and s1 is not None:
        if s0 < s1:
            raise ProactiveConfigError(
                f"tick_interval 必须满足 s0 >= s1，当前 s0={s0} < s1={s1}"
            )
        if s1 < 1:
            raise ProactiveConfigError(
                f"tick_interval_s1 必须 >= 1，当前值: {s1}"
            )


def _validate_profiles(user_profiles: dict[str, Any]) -> None:
    for name, values in user_profiles.items():
        if not isinstance(values, dict):
            raise ProactiveConfigError(
                f"proactive.profiles.{name} 必须是字典，"
                f"当前类型: {type(values).__name__}"
            )
        _validate_overrides(values)


def _validate_agent_keys(agent: dict[str, Any]) -> None:
    forbidden = set(agent.keys()) - _ALLOWED_AGENT_KEYS
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.agent 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(_ALLOWED_AGENT_KEYS))}"
        )


def _validate_drift_keys(drift: dict[str, Any]) -> None:
    forbidden = set(drift.keys()) - _ALLOWED_DRIFT_KEYS
    if forbidden:
        raise ProactiveConfigError(
            f"proactive.drift 出现非法键: {', '.join(sorted(forbidden))}。"
            f"允许键: {', '.join(sorted(_ALLOWED_DRIFT_KEYS))}"
        )


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _merge_profiles(user_profiles: dict[str, Any]) -> dict[str, Any]:
    """把用户自定义 profiles deep merge 到内置 PRESETS 上。"""
    merged: dict[str, Any] = copy.deepcopy(PRESETS)
    for name, values in user_profiles.items():
        current = merged.get(name)
        if isinstance(values, dict) and isinstance(current, dict):
            _deep_merge(current, values)
        else:
            merged[name] = copy.deepcopy(values)
    return merged


def _flatten_sections(preset: dict[str, Any]) -> dict[str, Any]:
    """把分节预设（trigger/gate/...）展平成单层 dict。"""
    flat: dict[str, Any] = {}
    for _section, values in preset.items():
        if isinstance(values, dict):
            flat.update(values)
    return flat


def load_proactive_config(p: dict[str, Any]) -> ProactiveConfig:
    """从配置字典加载 ProactiveConfig。

    Args:
        p: proactive 配置字典

    Returns:
        ProactiveConfig 实例

    Raises:
        ProactiveConfigError: 配置错误时抛出
    """
    if not isinstance(p, dict):
        raise ProactiveConfigError(
            f"proactive 配置必须是字典，当前类型: {type(p).__name__}"
        )
    _check_forbidden_root_keys(p)

    enabled = bool(p.get("enabled", False))

    # target 子块（或根级 default_channel/default_chat_id 兼容）
    target = p.get("target", {}) or {}
    if not isinstance(target, dict):
        raise ProactiveConfigError("proactive.target 必须是字典")
    default_channel = str(
        target.get("channel", p.get("default_channel", "telegram"))
    )
    default_chat_id = str(target.get("chat_id", ""))
    model = str(p.get("model", ""))

    # 预设名必填
    preset_name = p.get("profile")
    if not preset_name:
        raise ProactiveConfigError(
            "proactive.profile 是必填字段。"
            f"可选值: {', '.join(sorted(PRESETS))}"
        )
    user_profiles = p.get("profiles", {}) or {}
    if not isinstance(user_profiles, dict):
        raise ProactiveConfigError("proactive.profiles 必须是字典")
    _validate_profiles(user_profiles)
    profiles = _merge_profiles(user_profiles)
    if preset_name not in profiles:
        raise ProactiveConfigError(
            f"无效的 profile: '{preset_name}'。"
            f"只允许: {', '.join(sorted(profiles))}"
        )

    # overrides 白名单校验后合入
    overrides = p.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise ProactiveConfigError("proactive.overrides 必须是字典")
    _validate_overrides(overrides)
    preset = copy.deepcopy(profiles[preset_name])
    for section, values in overrides.items():
        preset[section].update(values)

    # agent / drift 子块校验
    agent = p.get("agent", {}) or {}
    if not isinstance(agent, dict):
        raise ProactiveConfigError("proactive.agent 必须是字典")
    _validate_agent_keys(agent)
    drift = p.get("drift", {}) or {}
    if not isinstance(drift, dict):
        raise ProactiveConfigError("proactive.drift 必须是字典")
    _validate_drift_keys(drift)

    # 展平预设 → 单层
    flat = _flatten_sections(preset)

    # 范围校验
    _validate_ranges(flat)

    # strategy params（内置）
    flat.update(STRATEGY_PARAMS)

    # agent/drift 子块映射到 agent_tick_* / drift_* 字段
    flat["agent_tick_model"] = str(agent.get("model", model))
    if "max_steps" in agent:
        flat["agent_tick_max_steps"] = int(agent["max_steps"])
    if "content_limit" in agent:
        flat["agent_tick_content_limit"] = int(agent["content_limit"])
    if "web_fetch_max_chars" in agent:
        flat["agent_tick_web_fetch_max_chars"] = int(agent["web_fetch_max_chars"])
    if "context_prob" in agent:
        flat["agent_tick_context_prob"] = float(agent["context_prob"])
    if "delivery_cooldown_hours" in agent:
        flat["agent_tick_delivery_cooldown_hours"] = int(
            agent["delivery_cooldown_hours"]
        )
    if "enabled" in drift:
        flat["drift_enabled"] = bool(drift["enabled"])
    if "max_steps" in drift:
        flat["drift_max_steps"] = int(drift["max_steps"])
    if "min_interval_hours" in drift:
        flat["drift_min_interval_hours"] = int(drift["min_interval_hours"])

    # 只保留 ProactiveConfig 已声明的字段，忽略多余键
    from initiative.proactive_config import config_field_names

    known = set(config_field_names())
    filtered = {k: v for k, v in flat.items() if k in known}

    return ProactiveConfig(
        enabled=enabled,
        default_channel=default_channel,
        default_chat_id=default_chat_id,
        model=model,
        **filtered,
    )
