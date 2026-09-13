"""笔记扩展 —— 持久化笔记 + 语义搜索。

相比初版内存版（重启丢失）：
  - 笔记存到 workspace/notes.jsonl
  - 每条笔记有 id / content / tags / created_at
  - 通过 VectorStore 提供语义搜索
  - 支持按 tag 过滤、关键词搜索、id 删除

工具集：
  - take_note(content, tags=[])：新建笔记
  - list_notes(tag=None, limit=50)：列出笔记
  - search_notes(query, top_k=5)：语义搜索（关键词 + TF-IDF 排序）
  - delete_note(note_id)：按 ID 删除
  - get_note(note_id)：按 ID 获取详情
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mind.tools import Tool, ToolRegistry
from mind.vector_store import VectorStore
from extensions.manager import Extension, ExtensionMeta

log = logging.getLogger(__name__)

NOTES_FILE = "notes.jsonl"


class _NoteStore:
    """JSONL 持久化存储 + 内存索引。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._file = workspace / NOTES_FILE
        self._notes: dict[str, dict[str, Any]] = {}
        self._vector = VectorStore()
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        for line in self._file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("笔记行解析失败: %s", exc)
                continue
            note_id = obj.get("id")
            if not note_id:
                continue
            self._notes[note_id] = obj
            self._vector.add(f"{obj.get('content', '')} {' '.join(obj.get('tags', []))}")

    def _persist(self) -> None:
        self._workspace.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as f:
            for note in self._notes.values():
                f.write(json.dumps(note, ensure_ascii=False) + "\n")

    def add(self, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        note_id = uuid.uuid4().hex[:12]
        # 用单调递增的 sequence 作为二级排序键
        seq = self._next_seq()
        note = {
            "id": note_id,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "_seq": seq,
        }
        self._notes[note_id] = note
        self._vector.add(f"{content} {' '.join(tags or [])}")
        self._persist()
        return note

    _seq_counter: int = 0

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def list(
        self,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        items = list(self._notes.values())
        if tag:
            items = [n for n in items if tag in n.get("tags", [])]
        # 用 _seq 排序：新增靠后 → 反序后靠前
        items.sort(key=lambda n: n.get("_seq", 0), reverse=True)
        return items[:limit]

    def get(self, note_id: str) -> dict[str, Any] | None:
        return self._notes.get(note_id)

    def delete(self, note_id: str) -> bool:
        note = self._notes.pop(note_id, None)
        if note is None:
            return False
        self._vector.clear()
        for n in self._notes.values():
            self._vector.add(f"{n.get('content', '')} {' '.join(n.get('tags', []))}")
        self._persist()
        return True

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict[str, Any], float]]:
        if not query.strip() or not self._notes:
            return []
        # 用 _seq 顺序的 notes 作为 vector 的并联索引
        ordered = sorted(self._notes.values(), key=lambda n: n.get("_seq", 0))
        if not ordered:
            return []
        results = self._vector.search(query, top_k=top_k * 2)
        scored: list[tuple[dict[str, Any], float]] = []
        for content, score in results:
            # 在 ordered 中通过 (content+tags) 唯一定位
            for note in ordered:
                if (
                    note.get("content") == content
                    or f"{note.get('content', '')} {' '.join(note.get('tags', []))}" == content
                ):
                    scored.append((note, score))
                    break
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._notes)


_store: _NoteStore | None = None


def _get_store() -> _NoteStore:
    global _store
    if _store is None:
        from mind.config import Config  # type: ignore[import-not-found]
        from pathlib import Path as _P

        workspace = _P.cwd() / "workspace"
        try:
            cfg = Config.from_dict_or_default()  # type: ignore[attr-defined]
            workspace = cfg.workspace
        except Exception:
            pass
        _store = _NoteStore(workspace)
    return _store


def _reset_store_for_test() -> None:
    """测试辅助：清空全局 store。"""
    global _store
    _store = None


def _set_store_for_test(store: _NoteStore) -> None:
    """测试辅助：注入 store。"""
    global _store
    _store = store


async def _tool_take_note(args: dict[str, Any]) -> str:
    content = (args.get("content") or "").strip()
    if not content:
        return "错误：缺少 content 参数"
    tags = args.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    note = _get_store().add(content, [str(t) for t in tags])
    return f"已记录笔记 #{note['id']}: {content}"


async def _tool_list_notes(args: dict[str, Any]) -> str:
    tag = args.get("tag")
    limit = int(args.get("limit") or 50)
    notes = _get_store().list(tag=tag, limit=limit)
    if not notes:
        return "当前没有笔记" if not tag else f"标签 {tag} 下没有笔记"
    lines = [f"[{n['id']}] {n['content']}" for n in notes]
    if tag:
        return f"标签 {tag} 下的笔记：\n" + "\n".join(lines)
    return f"共 {len(notes)} 条笔记：\n" + "\n".join(lines)


async def _tool_search_notes(args: dict[str, Any]) -> str:
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or 5)
    if not query:
        return "错误：缺少 query 参数"
    results = _get_store().search(query, top_k=top_k)
    if not results:
        return f"未找到与 '{query}' 相关的笔记"
    lines = [f"[{n['id']}] (score={s:.2f}) {n['content']}" for n, s in results]
    return f"搜索 '{query}' 找到 {len(results)} 条：\n" + "\n".join(lines)


async def _tool_delete_note(args: dict[str, Any]) -> str:
    note_id = (args.get("note_id") or "").strip()
    if not note_id:
        return "错误：缺少 note_id 参数"
    if _get_store().delete(note_id):
        return f"已删除笔记 #{note_id}"
    return f"未找到笔记 #{note_id}"


async def _tool_get_note(args: dict[str, Any]) -> str:
    note_id = (args.get("note_id") or "").strip()
    if not note_id:
        return "错误：缺少 note_id 参数"
    note = _get_store().get(note_id)
    if note is None:
        return f"未找到笔记 #{note_id}"
    return json.dumps(note, ensure_ascii=False, indent=2)


class NoteExtension(Extension):
    meta = ExtensionMeta(
        name="notes",
        description="持久化笔记工具，记录/列出/搜索/删除",
        version="0.2.0",
        author="ProactiveMind",
    )

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(Tool(
            name="take_note",
            description="记录一条笔记到本地文件",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "笔记内容"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "标签列表（可选）",
                    },
                },
                "required": ["content"],
            },
            func=_tool_take_note,
        ))
        registry.register(Tool(
            name="list_notes",
            description="列出已记录的笔记",
            parameters={
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "按标签过滤（可选）"},
                    "limit": {"type": "integer", "description": "返回条数上限"},
                },
            },
            func=_tool_list_notes,
        ))
        registry.register(Tool(
            name="search_notes",
            description="语义搜索笔记（TF-IDF 排序）",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回条数"},
                },
                "required": ["query"],
            },
            func=_tool_search_notes,
        ))
        registry.register(Tool(
            name="delete_note",
            description="按 ID 删除一条笔记",
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "笔记 ID"},
                },
                "required": ["note_id"],
            },
            func=_tool_delete_note,
        ))
        registry.register(Tool(
            name="get_note",
            description="按 ID 获取一条笔记的完整内容",
            parameters={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "笔记 ID"},
                },
                "required": ["note_id"],
            },
            func=_tool_get_note,
        ))


def create_extension() -> Extension:
    return NoteExtension()