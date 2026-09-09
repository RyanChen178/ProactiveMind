"""记忆系统 —— 基于文件的简单持久化记忆。

记忆存储在 workspace 下的 MEMORY.md 中，每行一条事实。
这是 MVP 版本，后续会升级为向量检索。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mind.embeddings import EmbeddingStore


class MemoryStore:
    """简单的文件记忆存储。"""

    def __init__(
        self,
        workspace: Path,
        embedding_store: "EmbeddingStore | None" = None,
    ) -> None:
        self._workspace = workspace
        self._file = workspace / "MEMORY.md"
        self._pending_file = workspace / "PENDING.md"
        self._embedding_store = embedding_store
        self._ensure_file()

    def _ensure_file(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._file.write_text(
                "# 长期记忆\n\n每行一条事实。\n\n",
                encoding="utf-8",
            )
        if not self._pending_file.exists():
            self._pending_file.write_text(
                "# 待归档记忆\n\n由对话 consolidation 自动提取，尚未合并到长期记忆。\n\n",
                encoding="utf-8",
            )

    def attach_embedding_store(self, store: "EmbeddingStore") -> None:
        """运行时挂载 EmbeddingStore（可选）。"""
        self._embedding_store = store

    def append(self, fact: str) -> None:
        """追加一条事实到记忆文件。"""
        with self._file.open("a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")
        if self._embedding_store is not None:
            # 预热缓存：append 后立即生成 embedding
            self._embedding_store.embed_text(fact)

    def append_pending(self, facts: list[str]) -> None:
        """将候选事实追加到待归档缓冲。"""

        if not facts:
            return
        with self._pending_file.open("a", encoding="utf-8") as f:
            for fact in facts:
                f.write(f"- {fact}\n")
        if self._embedding_store is not None:
            for fact in facts:
                self._embedding_store.embed_text(fact)

    def read_pending(self) -> list[str]:
        """读取待人工确认的候选事实。"""

        return self._read_facts(self._pending_file)

    def unpromoted_pending(self) -> list[str]:
        """读取尚未收录到长期记忆的候选事实。"""

        existing = {self._fact_key(fact) for fact in self._read_facts(self._file)}
        pending: list[str] = []
        for fact in self.read_pending():
            key = self._fact_key(fact)
            if key in existing:
                continue
            pending.append(fact)
            existing.add(key)
        return pending

    def promote_pending(self) -> list[str]:
        """将尚未收录的候选事实追加到长期记忆。"""

        promoted: list[str] = []
        for fact in self.unpromoted_pending():
            self.append(fact)
            promoted.append(fact)
        return promoted

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

    def semantic_recall(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.01,
    ) -> list[tuple[str, float]]:
        """语义检索：在长期记忆中找到与 query 最相似的事实。

        Returns:
            [(fact, score), ...] 按 score 降序，最多 top_k 个。
        """
        if self._embedding_store is None:
            # 未挂载 embedding 时退化到关键词搜索（带 score=1.0）
            hits = self.search(query)[:top_k]
            return [(h, 1.0) for h in hits]
        facts = self._read_facts(self._file)
        if not facts:
            return []
        return self._embedding_store.semantic_search(
            query, facts, top_k=top_k, threshold=threshold,
        )

    def read_all(self) -> str:
        """读取全部记忆内容。"""
        return self._file.read_text(encoding="utf-8")

    @staticmethod
    def _read_facts(path: Path) -> list[str]:
        """从 Markdown 列表中提取事实。"""

        return [
            line[2:].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("- ") and line[2:].strip()
        ]

    @staticmethod
    def _fact_key(fact: str) -> str:
        """生成用于去重的事实标识。"""

        return " ".join(fact.split()).casefold()
