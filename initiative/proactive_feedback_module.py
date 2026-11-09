"""主动推送反馈接线 —— 跨 tick 记录 + 统计快照接入管线。

设计思想（与 a20b45af plugins/proactive_feedback/plugin.py 对齐）：

  - 反馈发生在推送之后的下一次用户回复（跨 tick 时序），
    因此分两个组件：
      1. FeedbackRecorder —— 渠道层调用：
         用户回复到达时，与近期推送做引用/主题比对评分，
         结果落 feedback 库（insert_feedback），
         供 VAD 情绪与策略追踪消费
      2. FeedbackStatsModule —— 管线 prompt 阶段：
         每轮把近期反馈统计快照（engagement/deflection 率 +
         最近事件摘要 + 各类型计数）写进 frame.slots，
         供提示词组装与 judge 参考
  - FeedbackRecorder.record()：
      score_feedback(user_text, recent_proactive) →
      FeedbackEvent → insert_feedback → 返回评分结果
      （近期无推送 → 返回 None 不落库）
  - 统计模块读 FeedbackReporter.summary()，
    库不可用 → 静默跳过不阻断

复用 mind/feedback_scorer（评分）、mind/feedback_db（落库）、
mind/feedback_stats（聚合）。

ProactiveMind 命名（initiative/proactive_feedback_module.py）
独立于原仓库。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from initiative.proactive_frame import ProactiveFrame
from mind.feedback_db import FeedbackEvent, insert_feedback, open_db
from mind.feedback_scorer import FeedbackScore, score_feedback
from mind.feedback_stats import FeedbackReporter

logger = logging.getLogger(__name__)

SLOT_FEEDBACK_STATS = "proactive:feedback:stats"


def _as_text(item: Any) -> str:
    """推送条目（dict 或 str）→ 文本。"""
    if isinstance(item, dict):
        return str(item.get("content") or item.get("title") or "")
    return str(item)


class FeedbackRecorder:
    """跨 tick 反馈记录器：用户回复 → 评分 → 落库。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def record(
        self,
        *,
        session_key: str,
        user_message_id: str,
        assistant_message_id: str,
        user_text: str,
        assistant_text: str = "",
        proactive_message_id: str | None = None,
        recent_proactive: list[Any] | None = None,
        lag_seconds: int | None = None,
    ) -> FeedbackScore | None:
        """评分一次用户回复对近期推送的反馈。

        归因对象取最近一条推送，其余推送作为比对候选集。

        Returns:
            评分结果；近期无推送（无法归因）返回 None 不落库。
        """
        recent = [_as_text(item) for item in (recent_proactive or [])]
        recent = [t for t in recent if t]
        if not recent:
            return None

        primary = recent[-1]
        others = recent[:-1]
        score = score_feedback(
            primary,
            user_text,
            candidates=others,
        )
        try:
            conn = open_db(self._db_path)
        except Exception as exc:
            logger.warning("[feedback] 反馈库打开失败: %s", exc)
            return score  # 评分返回但落库失败不抛

        try:
            insert_feedback(conn, FeedbackEvent(
                session_key=session_key,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                proactive_message_id=proactive_message_id,
                feedback_type=score.feedback_type,
                confidence=score.confidence,
                pa_score=score.pa_score,
                pua_score=score.pua_score,
                lag_seconds=lag_seconds,
                candidate_count=score.candidate_count,
                matched_by=score.matched_by,
                reason=score.reason,
            ))
        except Exception as exc:
            logger.warning("[feedback] 反馈落库失败: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return score


class FeedbackStatsModule:
    """prompt 阶段：近期反馈统计快照写进 frame.slots。"""

    slot = "proactive.prompt.feedback_stats"
    phase = "proactive.prompt"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        stats = self.collect()
        if stats is not None:
            frame.slots[SLOT_FEEDBACK_STATS] = stats
        return frame

    def collect(self) -> dict[str, Any] | None:
        """聚合近期反馈统计；库不可用返回 None。"""
        try:
            reporter = FeedbackReporter.open(self._db_path)
        except Exception as exc:
            logger.warning("[feedback] 统计打开失败，跳过: %s", exc)
            return None
        try:
            summary = reporter.summary()
            latest = reporter.session_summary(
                _latest_session(summary.sessions)
            ) if summary.sessions else {}
            return {
                "total_events": summary.total_events,
                "by_type": summary.by_type,
                "engagement_rate": summary.engagement_rate(),
                "deflection_rate": summary.deflection_rate(),
                "session_count": len(summary.sessions),
                "latest_events": latest.get("latest", []),
                "generated_at": summary.generated_at,
            }
        except Exception as exc:
            logger.warning("[feedback] 统计聚合失败，跳过: %s", exc)
            return None
        finally:
            reporter.close()


def _latest_session(sessions: dict[str, int]) -> str:
    """取事件数最多的会话键（快照聚焦主要会话）。"""
    return max(sessions, key=sessions.get) if sessions else ""
