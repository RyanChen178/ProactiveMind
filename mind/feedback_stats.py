"""反馈统计聚合 —— 报告级 API。

将反馈评分与持久化组合成报告友好的汇总视图，供 Dashboard / CLI 复用。

设计思想：
  - 一个 FeedbackReporter 聚合某个 db 连接
  - 提供 summary() 返回全局统计 + per-session 列表
  - 提供 session_summary() 按会话聚合
  - 类型注解清晰，调用方不需要懂 SQL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mind.feedback_db import (
    FeedbackEvent,
    avg_metric_by_type,
    event_count_by_type,
    list_session_events,
    open_db,
)


@dataclass(frozen=True)
class FeedbackSummary:
    """全局反馈统计快照。"""

    total_events: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    avg_pa_score: dict[str, float] = field(default_factory=dict)
    avg_pua_score: dict[str, float] = field(default_factory=dict)
    avg_lag_seconds: dict[str, float] = field(default_factory=dict)
    sessions: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""

    def engagement_rate(self) -> float:
        """用户对主动推送有反馈的比例（按 explicit_quote + topic_follow 之和）。"""
        if self.total_events == 0:
            return 0.0
        engaged = self.by_type.get("explicit_quote", 0) + self.by_type.get(
            "topic_follow", 0
        )
        return round(engaged / self.total_events, 4)

    def deflection_rate(self) -> float:
        """用户切走话题的比例。"""
        if self.total_events == 0:
            return 0.0
        return round(self.by_type.get("no_topic_follow", 0) / self.total_events, 4)

    def as_dict(self) -> dict:
        return {
            "total_events": self.total_events,
            "by_type": self.by_type,
            "avg_pa_score": self.avg_pa_score,
            "avg_pua_score": self.avg_pua_score,
            "avg_lag_seconds": self.avg_lag_seconds,
            "sessions": self.sessions,
            "engagement_rate": self.engagement_rate(),
            "deflection_rate": self.deflection_rate(),
            "generated_at": self.generated_at,
        }


class FeedbackReporter:
    """基于 sqlite3.Connection 的统计聚合器。"""

    def __init__(self, conn) -> None:
        self._conn = conn

    @classmethod
    def open(cls, path) -> "FeedbackReporter":
        return cls(open_db(Path(path)))

    def summary(self) -> FeedbackSummary:
        by_type = event_count_by_type(self._conn)
        total = sum(by_type.values())
        session_rows = self._conn.execute(
            "SELECT session_key, COUNT(*) AS n FROM proactive_feedback_events "
            "GROUP BY session_key"
        ).fetchall()
        sessions = {row["session_key"]: int(row["n"]) for row in session_rows}
        return FeedbackSummary(
            total_events=total,
            by_type=by_type,
            avg_pa_score=avg_metric_by_type(self._conn, "pa_score"),
            avg_pua_score=avg_metric_by_type(self._conn, "pua_score"),
            avg_lag_seconds=avg_metric_by_type(self._conn, "lag_seconds"),
            sessions=sessions,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def session_summary(self, session_key: str, *, limit: int = 50) -> dict:
        events = list_session_events(self._conn, session_key, limit=limit)
        return {
            "session_key": session_key,
            "event_count": len(events),
            "by_type": event_count_by_type(self._conn, session_key=session_key),
            "latest": events[:5],
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass