"""向量记忆检索测试。"""

from __future__ import annotations

import unittest

from mind.vector_store import VectorStore, tokenize


class TokenizeTest(unittest.TestCase):
    def test_splits_english_words(self) -> None:
        tokens = tokenize("hello world")
        self.assertEqual(tokens, ["hello", "world"])

    def test_splits_chinese_chars(self) -> None:
        tokens = tokenize("你好世界")
        self.assertEqual(tokens, ["你", "好", "世", "界"])

    def test_mixed_chinese_english(self) -> None:
        tokens = tokenize("用户喜欢 python")
        self.assertIn("python", tokens)
        self.assertIn("用", tokens)
        self.assertIn("户", tokens)

    def test_ignores_single_chars_english(self) -> None:
        tokens = tokenize("a b c")
        self.assertEqual(tokens, [])

    def test_handles_empty_string(self) -> None:
        self.assertEqual(tokenize(""), [])


class VectorStoreTest(unittest.TestCase):
    def test_empty_store_returns_no_results(self) -> None:
        store = VectorStore()
        self.assertEqual(store.search("test"), [])
        self.assertEqual(store.size, 0)

    def test_add_increases_size(self) -> None:
        store = VectorStore()
        store.add("hello world")
        store.add("foo bar")
        self.assertEqual(store.size, 2)

    def test_search_returns_relevant_results(self) -> None:
        store = VectorStore()
        store.add("用户喜欢用 Python 编程")
        store.add("用户住在上海")
        store.add("用户养了一只猫")
        store.add("用户喜欢喝咖啡")

        results = store.search("编程")

        self.assertGreater(len(results), 0)
        self.assertIn("用户喜欢用 Python 编程", results[0][0])

    def test_search_returns_empty_for_no_match(self) -> None:
        store = VectorStore()
        store.add("用户喜欢猫")
        results = store.search("量子物理")
        self.assertEqual(results, [])

    def test_search_top_k_limits_results(self) -> None:
        store = VectorStore()
        for i in range(10):
            store.add(f"用户喜欢物品 {i}")
        results = store.search("用户", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_search_returns_score(self) -> None:
        store = VectorStore()
        store.add("hello world")
        results = store.search("hello", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][1], 0.0)

    def test_threshold_filters_low_scores(self) -> None:
        store = VectorStore()
        store.add("hello world")
        store.add("量子物理研究")
        results = store.search("hello", threshold=0.999)
        # 完全匹配的分数应该低于 0.999（因为 IDF 和 DF 因子）
        self.assertEqual(results, [])

    def test_clear_resets_store(self) -> None:
        store = VectorStore()
        store.add("test")
        store.clear()
        self.assertEqual(store.size, 0)
        self.assertEqual(store.search("test"), [])

    def test_rebuild_from_texts(self) -> None:
        store = VectorStore()
        store.rebuild(["item one", "item two", "item three"])
        self.assertEqual(store.size, 3)
        results = store.search("item")
        self.assertGreater(len(results), 0)

    def test_rebuild_replaces_old_entries(self) -> None:
        store = VectorStore()
        store.add("old entry")
        store.rebuild(["new entry"])
        self.assertEqual(store.size, 1)
        results = store.search("new")
        self.assertGreater(len(results), 0)

    def test_chinese_semantic_search(self) -> None:
        store = VectorStore()
        store.add("用户的生日是三月十五日")
        store.add("用户的职业是软件工程师")
        store.add("用户最喜欢的颜色是蓝色")

        results = store.search("生日")
        self.assertGreater(len(results), 0)
        self.assertIn("用户的生日是三月十五日", results[0][0])


if __name__ == "__main__":
    unittest.main()
