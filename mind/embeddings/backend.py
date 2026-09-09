"""Embedding 后端抽象 —— 本地 TF-IDF 与 HTTP 两种实现。

设计目的：
  - 提供统一的 embed_query / embed_corpus 接口
  - 本地 TFIDFBackend 复用现有 VectorStore，无需外部依赖
  - HTTPBackend 走 OpenAI-compatible embedding API（dashscope/openai）
  - 后端对调用方完全透明，切换只改配置
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import httpx

from mind.vector_store import VectorStore, tokenize

log = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Embedding 后端抽象基类。"""

    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """对单条 query 生成 embedding 向量。"""
        ...

    @abstractmethod
    def embed_corpus(self, texts: Sequence[str]) -> list[list[float]]:
        """对批量文档生成 embedding 矩阵（行=文档）。"""
        ...


class LocalTFIDFBackend(EmbeddingBackend):
    """本地 TF-IDF 后端 —— 不依赖外部 API。

    内部用 VectorStore 一次性构造 corpus 向量化；query 走临时 doc 路径，
    用与 corpus 相同的 token 频率反演 TF-IDF（共享 self._df / self._doc_count）。

    所有向量在维度上对齐：使用 corpus 全集 token 的固定排序作为坐标轴。
    """

    def __init__(self) -> None:
        self.name = "local-tfidf"
        self._vector_store = VectorStore()
        self._vocab: list[str] = []  # 固定排序的 token 列表

    def fit(self, corpus: Sequence[str]) -> None:
        """从语料构建 IDF 统计信息与共享词汇表。"""
        self._vector_store.rebuild(list(corpus))
        # 用 corpus 全集 token 作为固定词汇表
        vocab: set[str] = set()
        for text in corpus:
            for token in tokenize(text):
                vocab.add(token)
        self._vocab = sorted(vocab)

    def embed_query(self, text: str) -> list[float]:
        """生成与 corpus 维度一致的 TF-IDF 向量。"""
        tokens = tokenize(text)
        if not self._vocab:
            return [0.0] * 0
        if not tokens:
            return [0.0] * len(self._vocab)
        from collections import Counter

        tf = Counter(tokens)
        sparse = self._vector_store._tfidf_vector(tf)
        # 严格按 vocab 顺序展开，未出现的 token = 0
        return [sparse.get(token, 0.0) for token in self._vocab]

    def embed_corpus(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化语料。"""
        return [self.embed_query(t) for t in texts]

    @property
    def dim(self) -> int:
        return len(self._vocab)


class HTTPEmbeddingBackend(EmbeddingBackend):
    """HTTP 后端 —— 调用 OpenAI-compatible embedding API。

    配置：
      base_url = "https://api.openai.com/v1"
      model = "text-embedding-3-small"
      api_key = "sk-..."

    响应格式（OpenAI）：
      {"data": [{"embedding": [0.1, 0.2, ...]}, ...]}
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        dim: int = 0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not model:
            raise ValueError("model 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._dim = dim
        self.name = f"http:{model}"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        try:
            resp = await self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            log.warning("HTTP embedding 调用失败: %s", exc)
            return [[] for _ in texts]
        items = data.get("data") or []
        result: list[list[float]] = []
        for item in items:
            emb = item.get("embedding") if isinstance(item, dict) else None
            if isinstance(emb, list):
                result.append([float(x) for x in emb])
            else:
                result.append([])
        return result

    def embed_query(self, text: str) -> list[float]:
        """同步接口：用 asyncio.run 包装异步调用。

        不推荐在已有 event loop 的上下文中使用；生产场景应直接 await `_embed_async`。
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._embed_async([text]))[0]
        # 在 event loop 内只能降级
        log.warning("HTTP embedding 在已运行的 loop 内被同步调用，建议改用 await")
        return []

    def embed_corpus(self, texts: Sequence[str]) -> list[list[float]]:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._embed_async(list(texts)))
        return [[] for _ in texts]

    @property
    def dim(self) -> int:
        return self._dim


def _sparse_to_dense(sparse: dict[str, float]) -> list[float]:
    """把稀疏 TF-IDF dict 序列化为稠密 list。

    用排序后的 key 列表作为"维度"，这样同一 backend 下向量长度一致。
    """
    keys = sorted(sparse.keys())
    return [sparse[k] for k in keys]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度。维度不一致时取较短长度。"""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = sum(x * x for x in a[:n]) ** 0.5
    norm_b = sum(x * x for x in b[:n]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
