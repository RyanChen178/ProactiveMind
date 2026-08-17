"""Markdown 记忆存储测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.memory import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def test_promotes_only_new_pending_facts_and_keeps_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            memory = MemoryStore(workspace)
            memory.append("用户偏好 Python")
            memory.append_pending(
                [
                    "用户偏好  Python",
                    "用户维护 ProactiveMind 项目",
                    "用户维护 ProactiveMind 项目",
                ]
            )

            promoted = memory.promote_pending()

            self.assertEqual(promoted, ["用户维护 ProactiveMind 项目"])
            self.assertEqual(
                memory.search("ProactiveMind"), ["用户维护 ProactiveMind 项目"]
            )
            self.assertEqual(
                memory.read_pending(),
                [
                    "用户偏好  Python",
                    "用户维护 ProactiveMind 项目",
                    "用户维护 ProactiveMind 项目",
                ],
            )
            self.assertEqual(memory.unpromoted_pending(), [])
            self.assertEqual(memory.promote_pending(), [])


if __name__ == "__main__":
    unittest.main()
