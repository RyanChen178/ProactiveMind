"""情绪模块 —— VAD 状态接入主动链路 prompt 阶段。

设计思想（与 a20b45af plugins/emotion/plugin.py 中的
EmotionProactivePromptModule 对齐）：

  - 每轮 prompt 阶段读取当前 VAD 情绪状态（半衰期衰减后的
    快照），计算两个输出写入 frame.slots：
      1. proactive:prompt:system_bottom:emotion
         —— 附加到系统提示词底部的语气区块
         （如"用户当前情绪积极，可适度主动关怀"）
      2. proactive:effect:emotion
         —— 效应审计 dict（VAD 三维 / tone / 阈值调整 /
            final_threshold），供观测与策略追踪
  - 阈值调整机制：积极情绪小幅降门槛（更愿意推），
    消极情绪小幅升门槛（少打扰）—— compute_tick_effect
    已实现，本模块负责把 base_threshold 从 frame.slots
    读入（缺省 0.60）并把效应写回
  - 情绪数据库不可用（连接失败等）→ 静默跳过，
    不阻断管线

复用 mind/emotion_state 的 compute_tick_effect / EmotionState。

ProactiveMind 命名（initiative/proactive_emotion_module.py）
独立于原仓库。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from initiative.proactive_frame import ProactiveFrame
from mind.emotion_state import compute_tick_effect, open_db

logger = logging.getLogger(__name__)

SLOT_PROMPT_SECTION = "proactive:prompt:system_bottom:emotion"
SLOT_EFFECT = "proactive:effect:emotion"
SLOT_BASE_THRESHOLD = "proactive:base_judge_send_threshold"

DEFAULT_BASE_THRESHOLD = 0.60


class EmotionPromptModule:
    """prompt 阶段：VAD 情绪 → 语气区块 + 阈值调整效应。"""

    slot = "proactive.prompt.emotion"
    phase = "proactive.prompt"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        section, effect = self.build(frame)
        if section:
            frame.slots[SLOT_PROMPT_SECTION] = section
        if effect is not None:
            frame.slots[SLOT_EFFECT] = effect
        return frame

    def build(
        self, frame: ProactiveFrame
    ) -> tuple[str, dict[str, Any] | None]:
        """读取情绪状态并计算 (语气区块, 效应 dict)。

        数据库不可用或任何异常 → ("", None) 静默跳过。
        """
        try:
            conn = open_db(self._db_path)
        except Exception as exc:
            logger.warning("[emotion] 情绪库打开失败，跳过: %s", exc)
            return "", None

        try:
            base_threshold = _read_threshold(frame)
            effect = compute_tick_effect(
                conn,
                tick_id=f"frame:{frame.input.started_at.isoformat()}",
                session_key=str(frame.input.session_key),
                base_threshold=base_threshold,
                now=frame.input.started_at,
            )
        except Exception as exc:
            logger.warning("[emotion] 效应计算失败，跳过: %s", exc)
            return "", None
        finally:
            try:
                conn.close()
            except Exception:
                pass

        return effect.prompt_section, _effect_dict(effect, base_threshold)


def _read_threshold(frame: ProactiveFrame) -> float:
    """从 frame.slots 读基准阈值（缺省 0.60），越界裁到 [0, 1]。"""
    raw = frame.slots.get(SLOT_BASE_THRESHOLD)
    if raw is None:
        return DEFAULT_BASE_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BASE_THRESHOLD
    return max(0.0, min(1.0, value))


def _effect_dict(effect: Any, base_threshold: float) -> dict[str, Any]:
    """效应对象 → 审计 dict（含 base/final 阈值与调整量）。"""
    return {
        "provider_name": "emotion",
        "valence": effect.valence,
        "arousal": effect.arousal,
        "dominance": effect.dominance,
        "tone_label": effect.tone_label,
        "expected_effect": effect.expected_effect,
        "base_threshold": base_threshold,
        "final_threshold": effect.final_threshold,
        "threshold_delta": effect.threshold_delta,
    }
