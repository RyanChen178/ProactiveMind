"""主动链路投递模块 —— 决策结果的最终发送与落库。

设计思想（与 a20b45af proactive_v2/modules_deliver.py 对齐）：

  - deliver 阶段消费 resolve/judge 的决策结果：
      decision=send → 三步投递流水线
      decision=skip → 只记 tick 日志
  - 投递流水线（send 路径）：
      1. 去重闸：与近期已推送消息做雷同判断
         （复用 MessageDeduper；重复 → 降级为 skip，不发送）
      2. 发送：调用注入的 send_fn(text)；
         失败不落库（下轮可重试）
      3. 落库：ProactiveStateStore.record_delivery 记录本次投递
         （供 cooldown gate 与去重窗口查询）+ log_tick 审计
  - 每步结果写回 frame.slots（delivered / delivery_status /
    delivery_reason），供 schedule 与观测消费
  - 去重器 / 状态库 / 发送函数全部依赖注入（duck typing），
    单测可全部用假件

复用 initiative.proactive_deduper.MessageDeduper 与
initiative.proactive_state.ProactiveStateStore。

ProactiveMind 命名（initiative/proactive_deliver.py）独立于原仓库。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from initiative.proactive_frame import ProactiveFrame
from initiative.proactive_state import DeliveryRecord

logger = logging.getLogger(__name__)

SLOT_DECISION = "decision"  # "send" | "skip"
SLOT_DRAFT_MESSAGE = "draft_message"
SLOT_DELIVERED = "delivered"
SLOT_DELIVERY_STATUS = "delivery_status"  # sent / deduped / failed / skipped
SLOT_DELIVERY_REASON = "delivery_reason"

DECISION_SEND = "send"
DECISION_SKIP = "skip"


class ProactiveDeliverer:
    """投递执行器：去重闸 → 发送 → 落库。"""

    def __init__(
        self,
        *,
        session_key: str,
        send_fn: Callable[[str], Awaitable[bool]] | None,
        state_store: Any,
        deduper: Any = None,
        recent_proactive_fn: Callable[[], list[Any]] = lambda: [],
    ) -> None:
        self._session_key = session_key
        self._send_fn = send_fn
        self._state_store = state_store
        self._deduper = deduper
        self._recent_proactive_fn = recent_proactive_fn

    async def deliver(
        self,
        message: str,
        *,
        message_id: str = "",
        payload_kind: str | None = None,
        now=None,
    ) -> dict[str, Any]:
        """执行 send 路径的投递流水线。

        Returns:
            {"delivered": bool, "status": str, "reason": str}
        """
        text = str(message or "").strip()
        if not text:
            return {
                "delivered": False,
                "status": "failed",
                "reason": "empty message",
            }

        # 1. 去重闸
        if self._deduper is not None:
            try:
                is_dup, reason = await self._deduper.is_duplicate(
                    text,
                    recent_proactive=self._recent_proactive_fn(),
                )
            except Exception as exc:
                logger.warning("[deliver] 去重检查异常，放行: %s", exc)
                is_dup, reason = False, str(exc)
            if is_dup:
                logger.info("[deliver] 与近期推送雷同，跳过: %s", reason)
                self._log_tick("deduped", reason=reason)
                return {
                    "delivered": False,
                    "status": "deduped",
                    "reason": reason,
                }

        # 2. 发送
        if self._send_fn is None:
            logger.warning("[deliver] 未配置发送通道")
            return {
                "delivered": False,
                "status": "failed",
                "reason": "send_fn not configured",
            }
        try:
            ok = await self._send_fn(text)
        except Exception as exc:
            logger.warning("[deliver] 发送异常: %s", exc)
            ok = False
        if not ok:
            self._log_tick("failed", reason="send failed")
            return {
                "delivered": False,
                "status": "failed",
                "reason": "send failed",
            }

        # 3. 落库（发送成功才记录，供 cooldown 与去重查询）
        if self._state_store is not None:
            try:
                from datetime import datetime, timezone

                now_dt = now or datetime.now(timezone.utc)
                self._state_store.record_delivery(DeliveryRecord(
                    session_key=self._session_key,
                    message_id=message_id or text[:32],
                    pushed_at=now_dt,
                    payload_kind=payload_kind,
                    payload_preview=text[:80],
                ))
            except Exception as exc:
                logger.warning("[deliver] 投递落库失败（发送已完成）: %s", exc)

        self._log_tick("sent")
        return {"delivered": True, "status": "sent", "reason": ""}

    def _log_tick(self, action: str, *, reason: str = "") -> None:
        if self._state_store is None:
            return
        try:
            self._state_store.log_tick(
                self._session_key,
                action,
                note={"stage": "deliver", "reason": reason} if reason else None,
            )
        except Exception as exc:
            logger.warning("[deliver] tick 日志写入失败: %s", exc)


class ProactiveDeliverModule:
    """deliver 阶段的 phase 模块壳。"""

    slot = "deliver"
    phase = "proactive.deliver"

    def __init__(self, deliverer: ProactiveDeliverer) -> None:
        self._deliverer = deliverer

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        decision = str(frame.slots.get(SLOT_DECISION) or DECISION_SKIP)
        if decision != DECISION_SEND:
            frame.slots[SLOT_DELIVERED] = False
            frame.slots[SLOT_DELIVERY_STATUS] = "skipped"
            frame.slots[SLOT_DELIVERY_REASON] = f"decision={decision}"
            return frame

        message = str(frame.slots.get(SLOT_DRAFT_MESSAGE) or "")
        message_id = str(frame.slots.get("message_id") or "")
        result = await self._deliverer.deliver(
            message, message_id=message_id,
        )
        frame.slots[SLOT_DELIVERED] = result["delivered"]
        frame.slots[SLOT_DELIVERY_STATUS] = result["status"]
        frame.slots[SLOT_DELIVERY_REASON] = result["reason"]
        return frame
