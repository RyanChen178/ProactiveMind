"""Embedding 子系统：backend 抽象 + 持久化缓存 + 语义检索。"""

from mind.embeddings.backend import (
    EmbeddingBackend,
    HTTPEmbeddingBackend,
    LocalTFIDFBackend,
    cosine_similarity,
)
from mind.embeddings.store import (
    EmbeddedItem,
    EmbeddingStore,
    build_local_backend,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingStore",
    "EmbeddedItem",
    "HTTPEmbeddingBackend",
    "LocalTFIDFBackend",
    "build_local_backend",
    "cosine_similarity",
]