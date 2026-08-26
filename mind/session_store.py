"""SQLite 会话持久化。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SessionStore:
    """保存活动会话及其只追加消息。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        """创建最小会话 schema。"""

        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_call_id TEXT,
                    tool_calls_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS messages_session_id_id
                ON messages(session_id, id);
                """
            )

    def get_or_create_active_session(self) -> str:
        """返回活动会话；首次使用时创建。"""

        row = self._connection.execute(
            "SELECT value FROM session_state WHERE key = 'active_session_id'"
        ).fetchone()
        if row is None:
            return self.create_active_session()

        session_id = row["value"]
        exists = self._connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise RuntimeError(f"活动会话不存在: {session_id}")
        return session_id

    def create_active_session(self) -> str:
        """创建并切换到新的活动会话。"""

        session_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, created_at),
            )
            self._connection.execute(
                """
                INSERT INTO session_state (key, value)
                VALUES ('active_session_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (session_id,),
            )
        return session_id

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """向指定会话追加一条 OpenAI 格式消息。"""

        role = message.get("role")
        content = message.get("content")
        tool_call_id = message.get("tool_call_id")
        tool_calls = message.get("tool_calls")

        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("消息必须包含字符串 role 和 content")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise ValueError("tool_call_id 必须是字符串")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError("tool_calls 必须是列表")

        tool_calls_json = None
        if tool_calls is not None:
            tool_calls_json = json.dumps(tool_calls, ensure_ascii=False)

        with self._connection:
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            self._connection.execute(
                """
                INSERT INTO messages (
                    session_id, sequence, role, content, tool_call_id,
                    tool_calls_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    row["next_sequence"],
                    role,
                    content,
                    tool_call_id,
                    tool_calls_json,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """按追加顺序读取完整会话消息。"""

        rows = self._connection.execute(
            """
            SELECT id, role, content, tool_call_id, tool_calls_json
            FROM messages
            WHERE session_id = ?
            ORDER BY sequence
            """,
            (session_id,),
        ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            message: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["tool_call_id"] is not None:
                message["tool_call_id"] = row["tool_call_id"]
            if row["tool_calls_json"] is not None:
                try:
                    tool_calls = json.loads(row["tool_calls_json"])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"消息 {row['id']} 的 tool_calls_json 损坏"
                    ) from exc
                if not isinstance(tool_calls, list):
                    raise ValueError(
                        f"消息 {row['id']} 的 tool_calls_json 不是列表"
                    )
                message["tool_calls"] = tool_calls
            messages.append(message)
        return messages

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有会话及其消息数。"""
        rows = self._connection.execute(
            """
            SELECT s.id, s.created_at,
                   COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id, s.created_at
            ORDER BY s.created_at DESC
            """
        ).fetchall()

        active_id = self.get_or_create_active_session()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "message_count": row["message_count"],
                "last_message_at": row["last_message_at"],
                "is_active": row["id"] == active_id,
            }
            for row in rows
        ]

    def get_session_id_for_export(self, session_id: str) -> dict[str, Any] | None:
        """返回会话元信息（用于导出）。"""
        row = self._connection.execute(
            "SELECT id, created_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "created_at": row["created_at"]}

    def close(self) -> None:
        """关闭数据库连接。"""
        self._connection.close()
