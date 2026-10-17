"""主动推送评分 —— 决定本轮候选是否值得发。

设计思想（基于 a20b45af proactive_v2/judge.py 的多维评分思想）：

  - 四个评分维度（每个 [0, 1]）：
      energy_score       —— 当前活跃度（idle 越久越接近 1）
      recency_penalty    —— 近期推送过密则降分（与 cooldown gate 互补）
      context_quality    —— 上下文片段质量（外部可注入 score）
      interest_score     —— 用户/会话的兴趣度（外部可注入 score）
  - 加权汇总：weighted_sum / weight_total
  - 阈值过滤：score >= send_threshold 才建议发送
  - 不依赖 LLM，纯本地计算（async 留给将来扩展）
  - 异常/缺字段 → 退化默认值（不抛错）

依赖：所有维度与阈值都用 dataclass + duck typing
（任何有 .idle_minutes / .recent_deliveries / .quality / .interest 属性的对象都行）。

ProactiveMind 命名（initiative/proactive_judge.py）独立于原仓库。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# 默认权重（与 a20b45af 中典型配置同构）
DEFAULT_WEIGHTS: dict[str, float] = {
    "energy": 0.35,
    "recency": 0.15,
    "context": 0.25,
    "interest": 0.25,
}
DEFAULT_SEND_THRESHOLD = 0.60


@dataclass
class JudgeFeatures:
    """评分所需的原始特征（外部传入或由 sensor 派生）。"""

    idle_minutes: float = 0.0
    recent_delivery_count: int = 0
    context_quality: float | None = None
    interest_score: float | None = None


@dataclass
class JudgeResult:
    """评分结果。"""

    score: float
    decision: str  # "send" | "skip"
    per_dimension: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class _ScorerProvider(Protocol):
    """可选的 LLM 评分源（外部可注入；不注入时降级到本地默认）。"""

    def score_features(self, features: JudgeFeatures) -> JudgeFeatures: ...


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _energy_score(idle_minutes: float, idle_scale_minutes: float) -> float:
    """idle 越久 → 越接近 1（指数逼近）。"""
    if idle_scale_minutes <= 0:
        return 0.0
    return _clamp01(1.0 - math.exp(-idle_minutes / idle_scale_minutes))


def _recency_penalty(recent_delivery_count: int, cap: int) -> float:
    """近期推送数越多 → 越接近 0（推得太多就别再推了）。"""
    if cap <= 0:
        return 1.0
    return _clamp01(1.0 - recent_delivery_count / cap)


def _normalize(value: float | None, default: float) -> float:
    """None → 默认值（外部未提供）；越界裁到 [0, 1]。"""
    if value is None:
        return _clamp01(default)
    return _clamp01(value)


class ProactiveJudge:
    """主动推送评分器。"""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        send_threshold: float = DEFAULT_SEND_THRESHOLD,
        idle_scale_minutes: float = 15.0,
        recency_cap: int = 3,
    ) -> None:
        self._weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        self._send_threshold = float(send_threshold)
        self._idle_scale = float(idle_scale_minutes)
        self._recency_cap = int(recency_cap)

    def evaluate(self, features: JudgeFeatures) -> JudgeResult:
        """计算最终 score 并返回 send/skip 决策。"""
        per_dim: dict[str, float] = {}

        per_dim["energy"] = _energy_score(
            features.idle_minutes, self._idle_scale,
        )
        per_dim["recency"] = _recency_penalty(
            features.recent_delivery_count, self._recency_cap,
        )
        per_dim["context"] = _normalize(features.context_quality, default=0.5)
        per_dim["interest"] = _normalize(features.interest_score, default=0.5)

        total_weight = sum(self._weights.values())
        if total_weight <= 0:
            logger.warning("[proactive.judge] 权重全 0，使用均匀权")
            weight_items = [(k, 1.0) for k in per_dim]
        else:
            weight_items = [
                (k, self._weights.get(k, 0.0)) for k in per_dim
            ]

        score = sum(per_dim[k] * w for k, w in weight_items) / max(
            sum(w for _, w in weight_items), 1e-9
        )
        score = _clamp01(score)

        decision = "send" if score >= self._send_threshold else "skip"
        reason = self._explain(decision, per_dim, features)
        return JudgeResult(
            score=score,
            decision=decision,
            per_dimension=per_dim,
            reason=reason,
        )

    def _explain(
        self,
        decision: str,
        per_dim: dict[str, float],
        features: JudgeFeatures,
    ) -> str:
        energy = per_dim["energy"]
        recency = per_dim["recency"]
        weakest = min(per_dim, key=per_dim.get)
        bits = [
            f"energy={energy:.2f}",
            f"recency={recency:.2f}",
            f"weakest={weakest}({per_dim[weakest]:.2f})",
        ]
        if decision == "send":
            return f"通过 ({', '.join(bits)})"
        return f"低于阈值 ({', '.join(bits)})"

    def should_send(self, features: JudgeFeatures) -> tuple[bool, JudgeResult]:
        """便捷方法：返回 (是否发送, 完整结果)。"""
        result = self.evaluate(features)
        return result.decision == "send", result
