"""笔记插件 —— 演示插件如何注册工具。

在对话中可以通过 take_note / list_notes 工具管理简单笔记。
笔记存在内存中，进程结束后消失（演示用）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.tools import Tool, ToolRegistry
from plugins.manager import Plugin, PluginMeta

_notes: list[str] = []


async def _tool_take_note(args: dict[str, Any]) -> str:
    content = args.get("content", "").strip()
    if not content:
        return "错误：缺少 content 参数"
    timestamp = datetime.now().strftime("%H:%M")
    entry = f"[{timestamp}] {content}"
    _notes.append(entry)
    return f"已记录：{entry}"


async def _tool_list_notes(_: dict[str, Any]) -> str:
    if not _notes:
        return "当前没有笔记"
    return "\n".join(f"{i+1}. {n}" for i, n in enumerate(_notes))


class NotePlugin(Plugin):
    meta = PluginMeta(
        name="notes",
        description="简单笔记工具，可在对话中记录和查看笔记",
        version="0.1.0",
        author="ProactiveMind",
    )

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            Tool(
                name="take_note",
                description="记录一条笔记",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "笔记内容",
                        }
                    },
                    "required": ["content"],
                },
                func=_tool_take_note,
            )
        )
        registry.register(
            Tool(
                name="list_notes",
                description="列出所有已记录的笔记",
                parameters={"type": "object", "properties": {}},
                func=_tool_list_notes,
            )
        )


def create_plugin() -> Plugin:
    return NotePlugin()
