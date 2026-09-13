"""笔记扩展测试（持久化 + 搜索）。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from extensions import notes as notes_ext
from extensions.manager import Extension, ExtensionMeta
from mind.tools import Tool, ToolRegistry


class _FakeRegistry:
    """最小 ToolRegistry 替身：仅记录注册的工具。"""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool


def _run_tool(tool: Tool, args: dict) -> str:
    """同步执行 async tool（用于测试）。"""
    return asyncio.run(tool.execute(args))


class NoteStoreTest(unittest.TestCase):
    """_NoteStore 持久化与检索行为。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.store = notes_ext._NoteStore(self.workspace)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_creates_file(self) -> None:
        self.store.add("hello world", ["greeting"])
        self.assertTrue((self.workspace / "notes.jsonl").exists())

    def test_add_returns_metadata(self) -> None:
        note = self.store.add("buy milk")
        self.assertIn("id", note)
        self.assertEqual(note["content"], "buy milk")
        self.assertEqual(note["tags"], [])
        self.assertIn("created_at", note)

    def test_list_returns_added_notes(self) -> None:
        self.store.add("a")
        self.store.add("b", ["work"])
        notes = self.store.list()
        self.assertEqual(len(notes), 2)

    def test_list_filter_by_tag(self) -> None:
        self.store.add("a", ["work"])
        self.store.add("b", ["home"])
        notes = self.store.list(tag="work")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["content"], "a")

    def test_list_sorted_newest_first(self) -> None:
        n1 = self.store.add("first")
        n2 = self.store.add("second")
        notes = self.store.list()
        self.assertEqual(notes[0]["id"], n2["id"])
        self.assertEqual(notes[1]["id"], n1["id"])

    def test_list_respects_limit(self) -> None:
        for i in range(10):
            self.store.add(f"note {i}")
        notes = self.store.list(limit=3)
        self.assertEqual(len(notes), 3)

    def test_get_existing(self) -> None:
        note = self.store.add("x")
        got = self.store.get(note["id"])
        self.assertEqual(got["content"], "x")

    def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(self.store.get("nonexistent"))

    def test_delete_existing(self) -> None:
        note = self.store.add("to delete")
        self.assertTrue(self.store.delete(note["id"]))
        self.assertIsNone(self.store.get(note["id"]))

    def test_delete_missing(self) -> None:
        self.assertFalse(self.store.delete("nonexistent"))

    def test_search_finds_relevant(self) -> None:
        self.store.add("用户喜欢喝咖啡")
        self.store.add("项目用 Python 写")
        self.store.add("周末常去咖啡馆")
        results = self.store.search("咖啡", top_k=2)
        self.assertGreater(len(results), 0)
        # 至少一个含咖啡
        self.assertTrue(any("咖啡" in n["content"] for n, _ in results))

    def test_search_empty_query(self) -> None:
        self.store.add("x")
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("   "), [])

    def test_search_empty_store(self) -> None:
        self.assertEqual(self.store.search("anything"), [])

    def test_persistence_round_trip(self) -> None:
        """新 store 应能从现有 jsonl 文件恢复。"""
        note = self.store.add("persisted note", ["important"])
        # 新建 store 指向同一目录
        store2 = notes_ext._NoteStore(self.workspace)
        self.assertEqual(store2.size, 1)
        loaded = store2.get(note["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["content"], "persisted note")
        self.assertEqual(loaded["tags"], ["important"])

    def test_corrupted_line_skipped(self) -> None:
        """损坏的 jsonl 行应被跳过，不影响其它记录。"""
        self.store.add("valid")
        # 手动写入一行坏 JSON
        path = self.workspace / "notes.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps({"id": "manual", "content": "manual", "tags": []}) + "\n")

        store2 = notes_ext._NoteStore(self.workspace)
        self.assertEqual(store2.size, 2)


class NoteExtensionTest(unittest.TestCase):
    """Extension 注册的工具行为。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        self.store = notes_ext._NoteStore(self.workspace)
        notes_ext._set_store_for_test(self.store)
        self.registry = _FakeRegistry()
        self.ext = notes_ext.create_extension()
        self.ext.register_tools(self.registry)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        notes_ext._reset_store_for_test()
        self._tmp.cleanup()

    def test_registers_five_tools(self) -> None:
        expected = {"take_note", "list_notes", "search_notes", "delete_note", "get_note"}
        self.assertEqual(set(self.registry.tools.keys()), expected)

    def test_take_note_tool(self) -> None:
        result = _run_tool(self.registry.tools["take_note"], {
            "content": "test content",
            "tags": ["demo"],
        })
        self.assertIn("test content", result)
        self.assertEqual(self.store.size, 1)

    def test_take_note_without_tags(self) -> None:
        result = _run_tool(self.registry.tools["take_note"], {"content": "x"})
        self.assertIn("x", result)
        note = self.store.list()[0]
        self.assertEqual(note["tags"], [])

    def test_take_note_empty_content(self) -> None:
        result = _run_tool(self.registry.tools["take_note"], {"content": "  "})
        self.assertIn("错误", result)

    def test_list_notes_tool(self) -> None:
        _run_tool(self.registry.tools["take_note"], {"content": "a"})
        _run_tool(self.registry.tools["take_note"], {"content": "b"})
        result = _run_tool(self.registry.tools["list_notes"], {})
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("2 条", result)

    def test_list_notes_with_tag_filter(self) -> None:
        _run_tool(self.registry.tools["take_note"], {
            "content": "x", "tags": ["work"],
        })
        _run_tool(self.registry.tools["take_note"], {"content": "y"})
        result = _run_tool(self.registry.tools["list_notes"], {"tag": "work"})
        self.assertIn("x", result)
        self.assertNotIn("y", result)

    def test_list_notes_empty(self) -> None:
        result = _run_tool(self.registry.tools["list_notes"], {})
        self.assertIn("没有", result)

    def test_search_notes_tool(self) -> None:
        _run_tool(self.registry.tools["take_note"], {"content": "用户喜欢喝咖啡"})
        _run_tool(self.registry.tools["take_note"], {"content": "项目用 Python"})
        result = _run_tool(self.registry.tools["search_notes"], {
            "query": "咖啡", "top_k": 1,
        })
        self.assertIn("咖啡", result)

    def test_search_notes_empty_query(self) -> None:
        _run_tool(self.registry.tools["take_note"], {"content": "x"})
        result = _run_tool(self.registry.tools["search_notes"], {"query": ""})
        self.assertIn("错误", result)

    def test_search_notes_no_match(self) -> None:
        _run_tool(self.registry.tools["take_note"], {"content": "咖啡"})
        result = _run_tool(self.registry.tools["search_notes"], {
            "query": "完全不相关 qqqxyz",
        })
        self.assertIn("未找到", result)

    def test_delete_note_tool(self) -> None:
        result = _run_tool(self.registry.tools["take_note"], {"content": "x"})
        # 提取 ID
        note_id = self.store.list()[0]["id"]
        result = _run_tool(self.registry.tools["delete_note"], {"note_id": note_id})
        self.assertIn("已删除", result)
        self.assertEqual(self.store.size, 0)

    def test_delete_missing_note(self) -> None:
        result = _run_tool(self.registry.tools["delete_note"], {"note_id": "nope"})
        self.assertIn("未找到", result)

    def test_get_note_tool(self) -> None:
        _run_tool(self.registry.tools["take_note"], {
            "content": "details", "tags": ["t1"],
        })
        note_id = self.store.list()[0]["id"]
        result = _run_tool(self.registry.tools["get_note"], {"note_id": note_id})
        self.assertIn("details", result)
        self.assertIn("t1", result)

    def test_get_note_missing(self) -> None:
        result = _run_tool(self.registry.tools["get_note"], {"note_id": "missing"})
        self.assertIn("未找到", result)


class NoteExtensionMetaTest(unittest.TestCase):
    """Extension 元信息。"""

    def test_meta_present(self) -> None:
        ext = notes_ext.create_extension()
        self.assertEqual(ext.meta.name, "notes")
        self.assertEqual(ext.meta.version, "0.2.0")


if __name__ == "__main__":
    unittest.main()