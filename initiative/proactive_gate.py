"""主动推送准入 Gate —— 决定本轮 tick 是否能推送。

设计思想（与 a20b45af proactive_v2/modules_gate.py 中 ProactiveGateChain 对齐）：

  - 五个 gate 顺序检查，任一不过即 blocked：
      1. 目标 chat_id 非空
      2. 用户不处于被动忙状态（passive_busy）
      3. 投递冷却窗口（cooldown）内未重复推送
      4. 任何动作（anyaction）频控通过
      5. 上下文回退（context_as_fallback）随机概率
  - 通过所有 gate 才放行
  - 任何 gate 阻断返回 GateResult(blocked=True, reason=...)
  - context_as_fallback_open 走随机抽样，单独统计

依赖最小化：state_store / any_action_gate / rng 都接受 duck typing，
方便单测用 SimpleNamespace 注入。

ProactiveMind 命名（initiative/proactive_gate.py）独立于原仓库。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from initiative.proactive_tick_context import AgentTickContext

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Gate 决策结果。"""

    blocked: bool
    reason: str
    base_score: float | None = None
    context_as_fallback_open: bool = False


class _AnyActionGate(Protocol):
    def should_act(
        self,
        *,
        now_utc: datetime,
        last_user_at: datetime | None,
    ) -> tuple[bool, Any]: ...


class _StateStoreLike(Protocol):
    """state_store 最小协议：冷却窗口 + 上下文回退时间查询。"""

    def count_deliveries_in_window(
        self, session_key: str, hours: float,
    ) -> int: ...

    def get_last_context_only_at(
        self, session_key: str,
    ) -> datetime | None: ...

    def count_context_only_in_window(
        self, session_key: str, hours: float,
    ) -> int: ...


class ProactiveGateChain:
    """主动推送五段式准入链。"""

    def __init__(
        self,
        *,
        cfg: Any,
        session_key: str,
        state_store: _StateStoreLike,
        any_action_gate: _AnyActionGate | None = None,
        last_user_at_fn: Callable[[], datetime | None] = lambda: None,
        passive_busy_fn: Callable[[str], bool] | None = None,
        rng: Any = None,
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._state_store = state_store
        self._any_action_gate = any_action_gate
        self._last_user_at_fn = last_user_at_fn
        self._passive_busy_fn = passive_busy_fn
        # 缺省 random.Random 实例
        if rng is None:
            import random
            rng = random.Random()
        self._rng = rng

    def check(self, ctx: AgentTickContext) -> GateResult:
        """按顺序检查五个 gate；任一不过即 blocked。"""
        if not str(getattr(self._cfg, "default_chat_id", "") or "").strip():
            logger.debug("[proactive.gate] no chat_id -> blocked")
            return GateResult(blocked=True, reason="no_target", base_score=None)

        if self._passive_busy_fn and self._passive_busy_fn(self._session_key):
            logger.debug("[proactive.gate] passive_busy -> blocked")
            return GateResult(blocked=True, reason="busy", base_score=None)

        cooldown = float(
            getattr(self._cfg, "agent_tick_delivery_cooldown_hours", 1) or 0
        )
        try:
            recent_count = self._state_store.count_deliveries_in_window(
                self._session_key, cooldown,
            )
        except Exception as exc:
            logger.warning("[proactive.gate] count_deliveries 失败: %s", exc)
            recent_count = 0
        if recent_count > 0:
            logger.debug("[proactive.gate] cooldown -> blocked")
            return GateResult(blocked=True, reason="cooldown", base_score=None)

        if self._any_action_gate is not None:
            try:
                should_act, meta = self._any_action_gate.should_act(
                    now_utc=ctx.now_utc,
                    last_user_at=self._last_user_at_fn(),
                )
            except Exception as exc:
                logger.warning("[proactive.gate] anyaction 失败: %s", exc)
                should_act, meta = True, None
            if not should_act:
                logger.debug("[proactive.gate] anyaction -> blocked meta=%s", meta)
                return GateResult(blocked=True, reason="presence", base_score=None)

        fallback_open = self._context_fallback_open()
        return GateResult(
            blocked=False,
            reason="passed",
            base_score=None,
            context_as_fallback_open=fallback_open,
        )

    def _context_fallback_open(self) -> bool:
        """按 cfg.agent_tick_context_prob 概率抽样上下文回退窗口。"""
        prob = float(getattr(self._cfg, "agent_tick_context_prob", 0.0) or 0.0)
        if prob <= 0:
            return False
        return self._rng.random() < prob
