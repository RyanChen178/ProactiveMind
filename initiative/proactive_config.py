"""主动链路配置 —— 扁平 dataclass，预设 + 覆盖驱动。

设计思想（与 a20b45af proactive_v2/config.py 对齐）：

  - 一个扁平 ProactiveConfig dataclass 承载全部主动链路参数
  - 分四组字段：
      必填运行信息    —— enabled / channel / chat_id / model
      预设控制参数    —— trigger / gate / anyaction / safety / context
                        （由 presets + overrides 决定初值）
      策略内置参数    —— score_weight 等（不对外暴露）
      Agent Tick 参数 —— max_steps / content_limit / cooldown / drift
  - 大部分算法参数内置在策略里，用户只调预设和少量覆盖

ProactiveMind 命名（initiative/proactive_config.py）独立于原仓库。
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ProactiveConfig:
    """主动链路全部配置（扁平结构）。"""

    # 必填运行信息
    enabled: bool = False
    default_channel: str = "telegram"
    default_chat_id: str = ""
    model: str = ""

    # === 以下参数由预设 + 覆盖控制 ===

    # Trigger
    tick_interval_s0: int = 4800
    tick_interval_s1: int = 2400
    tick_jitter: float = 0.3

    # Gate
    judge_send_threshold: float = 0.60

    # AnyAction
    anyaction_enabled: bool = False
    anyaction_daily_max_actions: int = 24
    anyaction_min_interval_seconds: int = 300
    anyaction_probability_min: float = 0.03
    anyaction_probability_max: float = 0.45
    anyaction_idle_scale_minutes: float = 240.0
    anyaction_reset_hour_local: int = 12
    anyaction_timezone: str = "Asia/Shanghai"

    # Safety
    delivery_dedupe_hours: int = 24
    message_dedupe_recent_n: int = 5

    # Context
    context_only_daily_max: int = 1
    context_only_min_interval_hours: int = 12

    # === 策略内置参数（由 presets.STRATEGY_PARAMS 提供） ===
    score_weight_energy: float = 0.40
    message_dedupe_enabled: bool = True
    recent_chat_messages: int = 20
    interval_seconds: int = 1800

    # === Agent Tick ===
    agent_tick_max_steps: int = 35
    agent_tick_model: str = ""
    agent_tick_content_limit: int = 5
    agent_tick_web_fetch_max_chars: int = 8000
    agent_tick_context_prob: float = 0.03
    agent_tick_delivery_cooldown_hours: int = 1
    drift_enabled: bool = False
    drift_max_steps: int = 20
    drift_min_interval_hours: int = 3

    def replace(self, **changes: object) -> "ProactiveConfig":
        """返回替换部分字段后的新实例（dataclasses.replace 别名）。"""
        from dataclasses import replace as _replace

        return _replace(self, **changes)  # type: ignore[arg-type]


def config_field_names() -> list[str]:
    """返回全部字段名（测试与校验用）。"""
    return [f.name for f in fields(ProactiveConfig)]
