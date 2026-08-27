"""向量记忆检索 —— 基于 TF-IDF + 余弦相似度的语义搜索。

不依赖外部 embedding 模型，用纯 Python 实现 TF-IDF 向量化。
MemoryStore 的每条记忆被索引为向量，搜索时返回最相似的 top-k。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from collections import Counter


def tokenize(text: str) -> list[str]:
    """简单分词：中英文混合，中文按字、英文按词。"""
    tokens: list[str] = []
    # 英文单词
    for m in re.findall(r"[a-zA-Z]{2,}", text.lower()):
        tokens.append(m)
    # 中文字符（单字）
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
    return tokens


@dataclass
class VectorEntry:
    """一条记忆的向量索引条目。"""

    text: str
    tokens: list[str] = field(default_factory=list)
    tf: Counter = field(default_factory=Counter)
    norm: float = 0.0


class VectorStore:
    """向量存储与检索。"""

    def __init__(self) -> None:
        self._entries: list[VectorEntry] = []
        self._df: Counter = Counter()  # document frequency
        self._doc_count: int = 0

    def add(self, text: str) -> None:
        """添加一条记忆到索引。"""
        tokens = tokenize(text)
        tf = Counter(tokens)
        # 更新 DF
        for term in tf:
            self._df[term] += 1
        self._doc_count += 1
        # 预计算 TF-IDF 向量范数
        vec = self._tfidf_vector(tf)
        norm = math.sqrt(sum(v * v for v in vec.values()))
        self._entries.append(
            VectorEntry(text=text, tokens=tokens, tf=tf, norm=norm)
        )

    def search(self, query: str, top_k: int = 5, threshold: float = 0.01) -> list[tuple[str, float]]:
        """搜索与 query 最相似的记忆，返回 (text, score) 列表。"""
        if not self._entries:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_tf = Counter(query_tokens)
        query_vec = self._tfidf_vector(query_tf)
        query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if query_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for entry in self._entries:
            score = self._cosine(query_vec, query_norm, entry)
            if score >= threshold:
                scores.append((entry.text, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tfidf_vector(self, tf: Counter) -> dict[str, float]:
        """计算 TF-IDF 向量。"""
        vec: dict[str, float] = {}
        for term, count in tf.items():
            if self._doc_count > 0:
                idf = math.log((self._doc_count + 1) / (self._df.get(term, 0) + 1)) + 1
            else:
                idf = 1.0
            vec[term] = count * idf
        return vec

    def _cosine(self, query_vec: dict[str, float], query_norm: float, entry: VectorEntry) -> float:
        """计算余弦相似度。"""
        if entry.norm == 0 or query_norm == 0:
            return 0.0
        entry_vec = self._tfidf_vector(entry.tf)
        # 点积
        dot = sum(query_vec.get(t, 0) * entry_vec.get(t, 0) for t in query_vec)
        return dot / (query_norm * entry.norm)

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._df.clear()
        self._doc_count = 0

    def rebuild(self, texts: list[str]) -> None:
        """清空并重建整个索引。"""
        self.clear()
        for text in texts:
            self.add(text)
