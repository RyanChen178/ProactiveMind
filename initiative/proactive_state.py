"""主动链路 SQLite 状态持久化。

设计思想（与 a20b45af proactive_v2/state.py 对齐）：

  - 三类表：
      deliveries            —— 推送去重 + ack 跟踪（按 message_id + session）
      context_only_timestamps —— "仅作背景"类提示词注入的去重时间戳
      tick_log             —— 每轮 tick 调度结果（用于离线分析）
  - 单一连接 + RLock 串行化（跨线程安全）
  - WAL + NORMAL synchronous 平衡吞吐与持久性
  - close() 幂等；__del__ 兜底回收

ProactiveMind 命名（initiative/proactive_state.py）独立于原仓库。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DEFAULT_DB_NAME = "proactive_state.db"


@dataclass
class DeliveryRecord:
    """一条已推送的记录（去重与 ack）。"""

    session_key: str
    message_id: str
    pushed_at: datetime
    payload_kind: str | None = None
    payload_preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "message_id": self.message_id,
            "pushed_at": self.pushed_at.isoformat(),
            "payload_kind": self.payload_kind,
            "payload_preview": self.payload_preview,
        }


def _utcnow() -> datetime:
    """统一取 UTC 当前时间，便于测试 patch。"""
    return datetime.now(timezone.utc)


def _parse_iso(text: str) -> datetime | None:
    """解析 ISO 时间字符串，失败返回 None。"""
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None


class ProactiveStateStore:
    """主动链路本地状态持久化。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                message_id TEXT NOT NULL,
                pushed_at TEXT NOT NULL,
                payload_kind TEXT,
                payload_preview TEXT,
                UNIQUE(session_key, message_id)
            );

            CREATE TABLE IF NOT EXISTS context_only_timestamps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                UNIQUE(session_key, kind)
            );

            CREATE TABLE IF NOT EXISTS tick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                action TEXT NOT NULL,
                base_score REAL,
                pushed_message_id TEXT,
                note TEXT
            );
            """
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._db.close()
            except Exception:
                pass

    def __del__(self) -> None:  # noqa: D401
        try:
            self.close()
        except Exception:
            pass

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._db

    # ------------------------------------------------------------------
    # deliveries
    # ------------------------------------------------------------------

    def record_delivery(self, record: DeliveryRecord) -> bool:
        """记录一条推送；返回是否新插入（False 表示已存在 = 重复推送）。"""
        with self._tx() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO deliveries
                    (session_key, message_id, pushed_at, payload_kind, payload_preview)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.session_key,
                        record.message_id,
                        record.pushed_at.isoformat(),
                        record.payload_kind,
                        record.payload_preview,
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def recent_delivery_messages(
        self, session_key: str, limit: int = 5
    ) -> list[str]:
        """最近推送的 message_id 列表（用于去重）。"""
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT message_id FROM deliveries
                WHERE session_key = ?
                ORDER BY pushed_at DESC, id DESC
                LIMIT ?
                """,
                (session_key, limit),
            ).fetchall()
        return [row["message_id"] for row in rows]

    def last_delivery_at(self, session_key: str) -> datetime | None:
        """最近一次推送时间（按 session）。"""
        with self._tx() as conn:
            row = conn.execute(
                """
                SELECT pushed_at FROM deliveries
                WHERE session_key = ?
                ORDER BY pushed_at DESC, id DESC
                LIMIT 1
                """,
                (session_key,),
            ).fetchone()
        if row is None:
            return None
        return _parse_iso(row["pushed_at"])

    # ------------------------------------------------------------------
    # context_only_timestamps
    # ------------------------------------------------------------------

    def record_context_only(
        self, session_key: str, kind: str, when: datetime
    ) -> None:
        """记录"仅作背景"事件的时间戳（用于按 kind 去重）。"""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO context_only_timestamps
                (session_key, kind, occurred_at)
                VALUES (?, ?, ?)
                """,
                (session_key, kind, when.isoformat()),
            )
            conn.commit()

    def last_context_only_at(
        self, session_key: str, kind: str
    ) -> datetime | None:
        with self._tx() as conn:
            row = conn.execute(
                """
                SELECT occurred_at FROM context_only_timestamps
                WHERE session_key = ? AND kind = ?
                """,
                (session_key, kind),
            ).fetchone()
        if row is None:
            return None
        return _parse_iso(row["occurred_at"])

    # ------------------------------------------------------------------
    # tick_log
    # ------------------------------------------------------------------

    def log_tick(
        self,
        session_key: str,
        action: str,
        *,
        base_score: float | None = None,
        pushed_message_id: str | None = None,
        note: dict[str, Any] | None = None,
    ) -> None:
        """记录一轮 tick 结果（action: 'executed' / 'idle' / 'skipped'）。"""
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO tick_log
                (session_key, occurred_at, action, base_score, pushed_message_id, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_key,
                    _utcnow().isoformat(),
                    action,
                    base_score,
                    pushed_message_id,
                    json.dumps(note or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

    def recent_ticks(
        self, session_key: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tick_log
                WHERE session_key = ?
                ORDER BY occurred_at DESC, id DESC
                LIMIT ?
                """,
                (session_key, limit),
            ).fetchall()
        return [dict(row) for row in rows]
