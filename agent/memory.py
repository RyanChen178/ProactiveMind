"""记忆系统 —— 基于文件的简单持久化记忆。

记忆存储在 workspace 下的 MEMORY.md 中，每行一条事实。
这是 MVP 版本，后续会升级为向量检索。
"""

from __future__ import annotations

from pathlib import Path


class MemoryStore:
    """简单的文件记忆存储。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._file = workspace / "MEMORY.md"
        self._ensure_file()

    def _ensure_file(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._file.write_text(
                "# 长期记忆\n\n每行一条事实。\n\n",
                encoding="utf-8",
            )

    def append(self, fact: str) -> None:
        """追加一条事实到记忆文件。"""
        with self._file.open("a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")

    def search(self, keyword: str) -> list[str]:
        """按关键词搜索记忆，返回匹配的行。"""
        if not keyword:
            return []
        content = self._file.read_text(encoding="utf-8")
        lines = content.splitlines()
        return [
            line.lstrip("- ").strip()
            for line in lines
            if line.startswith("- ") and keyword.lower() in line.lower()
        ]

    def read_all(self) -> str:
        """读取全部记忆内容。"""
        return self._file.read_text(encoding="utf-8")
