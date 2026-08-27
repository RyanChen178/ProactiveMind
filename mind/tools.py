"""工具系统 —— 注册、搜索、执行。"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime
from typing import Any, Callable, Awaitable

from mind.provider import ToolCall
from mind.permission import ToolPermission, create_default_permission
from mind.vector_store import VectorStore


# 工具执行函数的类型：接收 dict 参数，返回 str
ToolFunc = Callable[[dict[str, Any]], Awaitable[str]]


class Tool:
    """一个可被 LLM 调用的工具。"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: ToolFunc,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._func = func

    async def execute(self, arguments: dict[str, Any]) -> str:
        return await self._func(arguments)

    def to_schema(self) -> dict:
        """转成 OpenAI function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self, permission: ToolPermission | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._permission = permission or create_default_permission()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"错误：未知工具 '{call.name}'"

        allowed, reason = self._permission.check(call.name, call.arguments)
        if not allowed:
            return f"权限拒绝：{reason}"

        try:
            return await tool.execute(call.arguments)
        except Exception as exc:
            return f"工具执行出错: {exc}"


# ─── 内置工具 ───


async def _tool_get_time(_: dict[str, Any]) -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def _tool_shell(args: dict[str, Any]) -> str:
    """执行 shell 命令，返回 stdout。"""
    command = args.get("command", "")
    if not command:
        return "错误：缺少 command 参数"
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        result = stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            result += f"\n[exit={proc.returncode}] {err}"
        return result or "(无输出)"
    except asyncio.TimeoutError:
        return "错误：命令执行超时（30s）"


def _create_memory_func(memory_store):
    """创建 memorize 工具函数，闭包持有 memory_store 引用。"""

    async def _tool_memorize(args: dict[str, Any]) -> str:
        fact = args.get("fact", "").strip()
        if not fact:
            return "错误：缺少 fact 参数"
        memory_store.append(fact)
        return f"已记住：{fact}"

    return _tool_memorize


def _create_recall_func(memory_store, vector_store: VectorStore | None = None):
    """创建 recall 工具函数。"""

    async def _tool_recall(args: dict[str, Any]) -> str:
        keyword = args.get("keyword", "").strip()
        if not keyword:
            return "错误：缺少 keyword 参数"

        # 优先使用向量语义搜索
        if vector_store is not None and vector_store.size > 0:
            results = vector_store.search(keyword, top_k=5)
            if results:
                lines = [f"- {text} (相关度: {score:.2f})" for text, score in results]
                return "\n".join(lines)

        # 回退到关键词匹配
        results = memory_store.search(keyword)
        if not results:
            return "没有找到相关记忆"
        return "\n".join(f"- {r}" for r in results)

    return _tool_recall


def build_core_tools(
    memory_store,
    permission: ToolPermission | None = None,
    vector_store: VectorStore | None = None,
) -> ToolRegistry:
    """构建默认工具集。"""
    registry = ToolRegistry(permission=permission)

    registry.register(
        Tool(
            name="get_time",
            description="获取当前日期和时间",
            parameters={"type": "object", "properties": {}},
            func=_tool_get_time,
        )
    )

    registry.register(
        Tool(
            name="shell",
            description="执行 shell 命令并返回输出。用于查看文件、运行脚本等。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的命令",
                    }
                },
                "required": ["command"],
            },
            func=_tool_shell,
        )
    )

    registry.register(
        Tool(
            name="memorize",
            description="将一条事实保存到长期记忆中，以便跨会话保留。",
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "要记住的事实内容",
                    }
                },
                "required": ["fact"],
            },
            func=_create_memory_func(memory_store),
        )
    )

    registry.register(
        Tool(
            name="recall",
            description="从长期记忆中按关键词检索相关内容。",
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["keyword"],
            },
            func=_create_recall_func(memory_store, vector_store),
        )
    )

    return registry
