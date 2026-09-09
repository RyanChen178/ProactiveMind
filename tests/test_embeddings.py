"""Embedding 子系统测试：backend / store / MemoryStore 集成。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mind.embeddings import (
    EmbeddingStore,
    HTTPEmbeddingBackend,
    LocalTFIDFBackend,
    build_local_backend,
    cosine_similarity,
)
from mind.embeddings.backend import _sparse_to_dense
from mind.memory import MemoryStore


class SparseToDenseTest(unittest.TestCase):
    """稀疏向量转稠密。"""

    def test_returns_sorted_keys(self) -> None:
        sparse = {"b": 0.5, "a": 0.3, "c": 0.1}
        dense = _sparse_to_dense(sparse)
        self.assertEqual(dense, [0.3, 0.5, 0.1])

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(_sparse_to_dense({}), [])


class CosineSimilarityTest(unittest.TestCase):
    """余弦相似度。"""

    def test_identical_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2], [-1, -2]), -1.0)

    def test_zero_vector_returns_zero(self) -> None:
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)

    def test_unequal_lengths_truncates(self) -> None:
        # 仅较短长度参与计算
        self.assertAlmostEqual(
            cosine_similarity([1, 2, 3], [1, 2]), 1.0
        )

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(cosine_similarity([], []), 0.0)


class LocalTFIDFBackendTest(unittest.TestCase):
    """本地 TF-IDF backend。"""

    def test_embed_query_returns_consistent_dim(self) -> None:
        backend = LocalTFIDFBackend()
        backend.fit([
            "用户喜欢喝咖啡",
            "项目用 Python 写",
            "咖啡馆常去的地点",
        ])
        v1 = backend.embed_query("咖啡")
        v2 = backend.embed_query("Python")
        # 同一 backend 下向量维度一致
        self.assertEqual(len(v1), len(v2))

    def test_embed_query_returns_empty_for_unseen(self) -> None:
        backend = LocalTFIDFBackend()
        backend.fit(["hello world"])
        # 空 query 返回 vocab 维度的全 0 向量
        vec = backend.embed_query("")
        self.assertEqual(vec, [0.0, 0.0])

    def test_embed_corpus_returns_per_item(self) -> None:
        backend = LocalTFIDFBackend()
        backend.fit(["a b c", "d e f", "a e"])
        result = backend.embed_corpus(["a b", "f"])
        self.assertEqual(len(result), 2)
        # 每个元素是 list[float]
        for vec in result:
            self.assertIsInstance(vec, list)


class HTTPEmbeddingBackendTest(unittest.TestCase):
    """HTTP backend 构造与同步调用。"""

    def test_requires_base_url(self) -> None:
        with self.assertRaises(ValueError):
            HTTPEmbeddingBackend(base_url="", model="m", api_key="k")

    def test_requires_model(self) -> None:
        with self.assertRaises(ValueError):
            HTTPEmbeddingBackend(base_url="https://x.com", model="", api_key="k")

    def test_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            HTTPEmbeddingBackend(base_url="https://x.com", model="m", api_key="")

    def test_name_includes_model(self) -> None:
        backend = HTTPEmbeddingBackend(
            base_url="https://x.com", model="text-embedding-3-small", api_key="k",
            http_client=MagicMock(),
        )
        self.assertIn("text-embedding-3-small", backend.name)

    def test_sync_embed_query_in_running_loop_returns_empty(self) -> None:
        backend = HTTPEmbeddingBackend(
            base_url="https://x.com", model="m", api_key="k",
            http_client=MagicMock(),
        )
        import asyncio

        async def _in_loop():
            return backend.embed_query("text")

        result = asyncio.run(_in_loop())
        # 在 event loop 内同步调用被降级
        self.assertEqual(result, [])


class BuildLocalBackendTest(unittest.TestCase):
    """便捷构造。"""

    def test_returns_local_backend(self) -> None:
        backend = build_local_backend()
        self.assertIsInstance(backend, LocalTFIDFBackend)


class EmbeddingStoreTest(unittest.TestCase):
    """EmbeddingStore 缓存层。"""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._temp_dir.name) / "embeddings.db"
        self.backend = LocalTFIDFBackend()
        self.backend.fit([
            "hello world",
            "foo bar",
            "用户喜欢喝咖啡",
            "咖啡馆是周末去的地方",
            "项目用 Python 写",
        ])
        self.store = EmbeddingStore(self.backend, self._db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._temp_dir.cleanup()

    def test_embed_text_caches_in_memory(self) -> None:
        v1 = self.store.embed_text("hello")
        v2 = self.store.embed_text("hello")
        self.assertEqual(v1, v2)
        # 内存缓存大小应增加
        stats = self.store.stats()
        self.assertGreaterEqual(stats["memory_cache_size"], 1)

    def test_embed_text_persists_to_db(self) -> None:
        self.store.embed_text("hello world")
        stats = self.store.stats()
        self.assertEqual(stats["cached_entries"], 1)

    def test_persisted_cache_reused_across_instances(self) -> None:
        """新建 store 时应能复用之前的 SQLite 缓存。"""
        self.store.embed_text("hello world")
        self.store.close()

        # 用新 backend 但相同 db 重新打开
        new_backend = LocalTFIDFBackend()
        new_backend.fit(["hello world", "foo bar"])
        new_store = EmbeddingStore(new_backend, self._db_path)
        try:
            stats = new_store.stats()
            self.assertEqual(stats["cached_entries"], 1)
        finally:
            new_store.close()

    def test_embed_corpus_returns_per_text(self) -> None:
        result = self.store.embed_corpus(["hello", "world", "foo"])
        self.assertEqual(len(result), 3)

    def test_semantic_search_ranks_relevant_first(self) -> None:
        candidates = [
            "用户喜欢喝咖啡",
            "项目使用 Python 编程",
            "咖啡馆是周末去的地方",
            "数据库用 SQLite",
        ]
        results = self.store.semantic_search("咖啡偏好", candidates, top_k=3)
        self.assertGreater(len(results), 0)
        # "用户喜欢喝咖啡" 应该排在前 2
        top_texts = [r[0] for r in results[:2]]
        self.assertTrue(
            any("咖啡" in t for t in top_texts),
            f"expected 咖啡-related in top results, got {top_texts}",
        )

    def test_semantic_search_empty_candidates(self) -> None:
        results = self.store.semantic_search("query", [])
        self.assertEqual(results, [])

    def test_semantic_search_respects_threshold(self) -> None:
        # query 完全用 corpus 中没出现过的英文单词
        candidates = ["完全不相关的内容 xyz123"]
        results = self.store.semantic_search("apple banana orange", candidates, threshold=0.99)
        # 高阈值下应过滤掉低分结果（中文候选 vs 英文 query 几乎无重叠）
        self.assertEqual(results, [])

    def test_clear_removes_all(self) -> None:
        self.store.embed_text("hello")
        self.store.embed_text("world")
        self.store.clear()
        stats = self.store.stats()
        self.assertEqual(stats["cached_entries"], 0)
        self.assertEqual(stats["memory_cache_size"], 0)


class MemoryStoreSemanticTest(unittest.TestCase):
    """MemoryStore 集成语义检索。"""

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._workspace = Path(self._temp_dir.name)
        self.backend = LocalTFIDFBackend()
        self.backend.fit([
            "用户喜欢喝咖啡",
            "项目用 Python",
            "周末去咖啡馆",
            "咖啡馆常去的地点",
            "Python 编程语言",
        ])
        self._db_path = self._workspace / "embeddings.db"
        self.store = EmbeddingStore(self.backend, self._db_path)
        self.memory = MemoryStore(self._workspace, embedding_store=self.store)

    def tearDown(self) -> None:
        self.memory._embedding_store = None  # type: ignore[attr-defined]
        self.store.close()
        self._temp_dir.cleanup()

    def test_attach_embedding_store(self) -> None:
        # 默认构造不应挂载
        bare = MemoryStore(self._workspace)
        self.assertIsNone(bare._embedding_store)  # type: ignore[attr-defined]

    def test_append_warms_cache(self) -> None:
        self.memory.append("用户每天早上喝拿铁")
        stats = self.store.stats()
        self.assertGreaterEqual(stats["cached_entries"], 1)

    def test_semantic_recall_returns_relevant_facts(self) -> None:
        self.memory.append("用户喜欢喝咖啡")
        self.memory.append("项目用 Python 写")
        self.memory.append("周末常去咖啡馆")
        results = self.memory.semantic_recall("咖啡相关", top_k=2)
        self.assertGreater(len(results), 0)
        # 至少有一个含"咖啡"
        self.assertTrue(any("咖啡" in r[0] for r in results))

    def test_semantic_recall_falls_back_without_embedding(self) -> None:
        bare = MemoryStore(self._workspace)
        bare.append("hello world")
        results = bare.semantic_recall("hello", top_k=3)
        # 降级到关键词搜索
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "hello world")
        self.assertEqual(results[0][1], 1.0)

    def test_semantic_recall_empty_memory(self) -> None:
        results = self.memory.semantic_recall("anything")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()