"""主动推送反馈事件 SQLite 持久化。

设计思想（与参考项目 proactive_feedback/db 对齐）：

  - 单表 proactive_feedback_events 存储每条用户反馈事件
  - UNIQUE(user_message_id, proactive_message_id) 约束保证事件去重
  - 复合索引：session_key+created_at（按会话拉时间序列）、
    proactive_message_id（按消息查反馈）
  - WAL 模式 + NORMAL synchronous，平衡写入吞吐与持久性
  - insert_feedback 先删同 user_message_id 的旧记录（同用户消息只保留最近一次反馈）

ProactiveMind 路径：workspace/feedback/feedback.db
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeedbackEvent:
    """一条主动推送反馈事件。"""

    session_key: str
    user_message_id: str
    assistant_message_id: str
    proactive_message_id: str | None
    feedback_type: str
    confidence: str
    pa_score: float | None
    pua_score: float | None
    lag_seconds: int | None
    candidate_count: int
    matched_by: str
    reason: str


def open_db(path: Path) -> sqlite3.Connection:
    """初始化 SQLite 数据库，返回连接。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS proactive_feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            session_key TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            assistant_message_id TEXT NOT NULL,
            proactive_message_id TEXT,
            feedback_type TEXT NOT NULL,
            confidence TEXT NOT NULL,
            pa_score REAL,
            pua_score REAL,
            lag_seconds INTEGER,
            candidate_count INTEGER NOT NULL,
            matched_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(user_message_id, proactive_message_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pfe_session_created
        ON proactive_feedback_events(session_key, created_at);

        CREATE INDEX IF NOT EXISTS idx_pfe_proactive
        ON proactive_feedback_events(proactive_message_id);
        """
    )
    conn.commit()
    return conn


def insert_feedback(conn: sqlite3.Connection, event: FeedbackEvent) -> int:
    """插入一条反馈事件；同 user_message_id 旧记录被删除以保证唯一。

    Returns:
        新插入行的 rowid。
    """
    conn.execute(
        "DELETE FROM proactive_feedback_events WHERE user_message_id = ?",
        (event.user_message_id,),
    )
    cursor = conn.execute(
        """
        INSERT INTO proactive_feedback_events (
            session_key,
            user_message_id,
            assistant_message_id,
            proactive_message_id,
            feedback_type,
            confidence,
            pa_score,
            pua_score,
            lag_seconds,
            candidate_count,
            matched_by,
            reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.session_key,
            event.user_message_id,
            event.assistant_message_id,
            event.proactive_message_id,
            event.feedback_type,
            event.confidence,
            event.pa_score,
            event.pua_score,
            event.lag_seconds,
            event.candidate_count,
            event.matched_by,
            event.reason,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def list_session_events(
    conn: sqlite3.Connection,
    session_key: str,
    *,
    limit: int = 100,
) -> list[dict]:
    """按时间倒序列出某会话的全部反馈事件。"""
    rows = conn.execute(
        """
        SELECT * FROM proactive_feedback_events
        WHERE session_key = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (session_key, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def events_for_proactive(
    conn: sqlite3.Connection,
    proactive_message_id: str,
) -> list[dict]:
    """查询某条主动推送消息关联的全部反馈。"""
    rows = conn.execute(
        """
        SELECT * FROM proactive_feedback_events
        WHERE proactive_message_id = ?
        ORDER BY created_at ASC
        """,
        (proactive_message_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def delete_session_events(conn: sqlite3.Connection, session_key: str) -> int:
    """删除某会话的全部反馈事件（清理 / GDPR）。"""
    cursor = conn.execute(
        "DELETE FROM proactive_feedback_events WHERE session_key = ?",
        (session_key,),
    )
    conn.commit()
    return cursor.rowcount


def event_count_by_type(
    conn: sqlite3.Connection,
    *,
    session_key: str | None = None,
) -> dict[str, int]:
    """按 feedback_type 聚合事件数（用于统计面板）。"""
    if session_key is None:
        rows = conn.execute(
            "SELECT feedback_type, COUNT(*) AS n FROM proactive_feedback_events "
            "GROUP BY feedback_type"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT feedback_type, COUNT(*) AS n FROM proactive_feedback_events "
            "WHERE session_key = ? GROUP BY feedback_type",
            (session_key,),
        ).fetchall()
    return {row["feedback_type"]: int(row["n"]) for row in rows}


def avg_metric_by_type(
    conn: sqlite3.Connection,
    metric: str,
    *,
    session_key: str | None = None,
) -> dict[str, float]:
    """按 feedback_type 求某指标（pa_score / pua_score / lag_seconds）的平均值。"""
    if metric not in {"pa_score", "pua_score", "lag_seconds"}:
        raise ValueError(f"unsupported metric: {metric}")
    if session_key is None:
        rows = conn.execute(
            f"SELECT feedback_type, AVG({metric}) AS v "
            f"FROM proactive_feedback_events GROUP BY feedback_type"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT feedback_type, AVG({metric}) AS v "
            f"FROM proactive_feedback_events WHERE session_key = ? GROUP BY feedback_type",
            (session_key,),
        ).fetchall()
    return {row["feedback_type"]: float(row["v"] or 0.0) for row in rows}