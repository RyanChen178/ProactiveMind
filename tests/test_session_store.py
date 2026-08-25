"""SQLite 会话存储测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.session import Session
from agent.session_store import SessionStore


class SessionStoreTest(unittest.TestCase):
    def test_restores_active_session_messages_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "sessions.db"
            store = SessionStore(db_path)
            session_id = store.get_or_create_active_session()
            session = Session(
                persist_message=lambda message: store.append_message(
                    session_id, message
                )
            )
            session.add_user("现在几点？")
            session.add_assistant(
                "",
                tool_calls=[
                    {
                        "id": "call-time",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": "{}",
                        },
                    }
                ],
            )
            session.add_tool_result("call-time", "2026-08-10 10:00:00")
            session.add_assistant("现在是十点。")
            store.close()

            reopened = SessionStore(db_path)
            self.assertEqual(reopened.get_or_create_active_session(), session_id)
            self.assertEqual(
                reopened.load_messages(session_id),
                session.messages,
            )
            reopened.close()

            connection = sqlite3.connect(db_path)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            sequences = connection.execute(
                "SELECT sequence FROM messages ORDER BY sequence"
            ).fetchall()
            connection.close()
            self.assertEqual(integrity, ("ok",))
            self.assertEqual(sequences, [(1,), (2,), (3,), (4,)])

    def test_new_active_session_preserves_old_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.db")
            old_session_id = store.get_or_create_active_session()
            store.append_message(
                old_session_id,
                {"role": "user", "content": "第一段对话"},
            )

            new_session_id = store.create_active_session()

            self.assertNotEqual(new_session_id, old_session_id)
            self.assertEqual(store.get_or_create_active_session(), new_session_id)
            self.assertEqual(store.load_messages(new_session_id), [])
            self.assertEqual(
                store.load_messages(old_session_id),
                [{"role": "user", "content": "第一段对话"}],
            )
            store.close()


class SessionStoreListTest(unittest.TestCase):
    def test_list_sessions_returns_all_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.db")
            s1 = store.get_or_create_active_session()
            store.append_message(s1, {"role": "user", "content": "你好"})
            store.append_message(s1, {"role": "assistant", "content": "你好呀"})

            s2 = store.create_active_session()
            store.append_message(s2, {"role": "user", "content": "在吗"})

            sessions = store.list_sessions()

            self.assertEqual(len(sessions), 2)
            active = [s for s in sessions if s["is_active"]]
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["id"], s2)

            counts = {s["id"]: s["message_count"] for s in sessions}
            self.assertEqual(counts[s1], 2)
            self.assertEqual(counts[s2], 1)
            store.close()

    def test_list_sessions_empty_when_no_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.db")
            store.create_active_session()
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["message_count"], 0)
            store.close()

    def test_get_session_id_for_export_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.db")
            self.assertIsNone(store.get_session_id_for_export("nonexistent"))
            store.close()


if __name__ == "__main__":
    unittest.main()
