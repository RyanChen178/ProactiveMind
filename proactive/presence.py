"""主动推送 —— 用户活跃状态追踪。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, UTC


class PresenceStore:
    """记录用户最后活跃时间，供主动推送电量模型使用。"""

    def __init__(self, db_path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS presence ("
            "  key TEXT PRIMARY KEY,"
            "  last_user_at TEXT,"
            "  last_proactive_at TEXT"
            ")"
        )
        self._conn.commit()

    def record_user_message(self, now: datetime | None = None) -> None:
        """用户发消息时更新心跳。"""
        ts = (now or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO presence (key, last_user_at) VALUES ('default', ?) "
            "ON CONFLICT(key) DO UPDATE SET last_user_at = excluded.last_user_at",
            (ts,),
        )
        self._conn.commit()

    def record_proactive(self, now: datetime | None = None) -> None:
        """主动推送发送后记录时间。"""
        ts = (now or datetime.now(UTC)).isoformat()
        self._conn.execute(
            "INSERT INTO presence (key, last_proactive_at) VALUES ('default', ?) "
            "ON CONFLICT(key) DO UPDATE SET last_proactive_at = excluded.last_proactive_at",
            (ts,),
        )
        self._conn.commit()

    def get_last_user_at(self) -> datetime | None:
        """返回用户最后活跃时间。"""
        row = self._conn.execute(
            "SELECT last_user_at FROM presence WHERE key = 'default'"
        ).fetchone()
        if row is None or row["last_user_at"] is None:
            return None
        return datetime.fromisoformat(row["last_user_at"])

    def get_last_proactive_at(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT last_proactive_at FROM presence WHERE key = 'default'"
        ).fetchone()
        if row is None or row["last_proactive_at"] is None:
            return None
        return datetime.fromisoformat(row["last_proactive_at"])

    def close(self) -> None:
        self._conn.close()
