"""EmbeddingStore —— 记忆文本 + 向量的持久化缓存。

设计要点：
  - 按内容 hash 作为缓存 key，避免重复计算 embedding
  - 持久化到 SQLite，支持跨进程复用
  - 提供语义检索接口 semantic_search(query, top_k)
  - 与本地 backend / HTTP backend 都兼容
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from mind.embeddings.backend import (
    EmbeddingBackend,
    LocalTFIDFBackend,
    cosine_similarity,
)

log = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """文本内容的稳定哈希，用作缓存 key。"""
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddedItem:
    """缓存中的一条 embedding 条目。"""

    text: str
    embedding: list[float]
    content_hash: str
    created_at: str


class EmbeddingStore:
    """embedding 持久化缓存。"""

    def __init__(
        self,
        backend: EmbeddingBackend,
        db_path: Path,
    ) -> None:
        self._backend = backend
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                backend_name TEXT NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_embeddings_backend
                ON embeddings (backend_name);
            """
        )
        self._conn.commit()
        self._memory_cache: dict[str, EmbeddedItem] = {}

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def embed_text(self, text: str) -> list[float]:
        """获取文本的 embedding，命中缓存则直接返回。"""
        h = _content_hash(text)
        cached = self._memory_cache.get(h)
        if cached is not None:
            return cached.embedding
        with self._lock:
            row = self._conn.execute(
                """
                SELECT vector_json FROM embeddings
                WHERE content_hash = ? AND backend_name = ?
                """,
                (h, self._backend.name),
            ).fetchone()
            if row is not None:
                vec = json.loads(row["vector_json"])
                self._memory_cache[h] = EmbeddedItem(
                    text=text, embedding=vec, content_hash=h,
                    created_at="",
                )
                return vec

        # 缓存未命中，调后端计算
        vec = self._backend.embed_query(text)
        self._persist(h, text, vec)
        return vec

    def embed_corpus(self, texts: Sequence[str]) -> list[list[float]]:
        """批量获取 embedding，逐条过缓存。"""
        return [self.embed_text(t) for t in texts]

    def _persist(self, h: str, text: str, vec: list[float]) -> None:
        """把 embedding 写入数据库 + 内存缓存。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (content_hash, backend_name, text, vector_json, dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (h, self._backend.name, text, json.dumps(vec), len(vec), now),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                log.warning("embedding 持久化失败: %s", exc)
        self._memory_cache[h] = EmbeddedItem(
            text=text, embedding=vec, content_hash=h, created_at=now,
        )

    def semantic_search(
        self,
        query: str,
        candidates: Sequence[str],
        top_k: int = 5,
        threshold: float = 0.01,
    ) -> list[tuple[str, float]]:
        """在 candidates 中检索与 query 语义最相似的条目。

        Returns:
            [(text, score), ...] 按 score 降序，最多 top_k 个。
        """
        if not candidates:
            return []
        query_vec = self.embed_text(query)
        if not query_vec:
            return []

        scored: list[tuple[str, float]] = []
        for text in candidates:
            vec = self.embed_text(text)
            score = cosine_similarity(query_vec, vec)
            if score >= threshold:
                scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()
        self._memory_cache.clear()

    def stats(self) -> dict[str, Any]:
        """返回缓存统计信息。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM embeddings WHERE backend_name = ?",
                (self._backend.name,),
            ).fetchone()
        return {
            "backend": self._backend.name,
            "db_path": str(self._db_path),
            "cached_entries": row["n"] if row else 0,
            "memory_cache_size": len(self._memory_cache),
        }


def build_local_backend() -> EmbeddingBackend:
    """构造默认本地 TF-IDF backend（不依赖外部 API）。"""
    return LocalTFIDFBackend()