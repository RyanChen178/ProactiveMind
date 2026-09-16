"""MCP Client 测试。

用 Python 子进程模拟一个最小可用的 MCP server（stdin 收 JSON-RPC，
stdout 发 JSON-RPC 响应）。覆盖：
  - initialize + initialized 握手
  - tools/list 列出工具
  - tools/call 调用工具
  - 错误响应传播
  - 子进程管理（启动、关闭）
  - McpToolWrapper 包装
  - McpRegistry 聚合
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mind.mcp import (
    McpClient,
    McpError,
    McpRegistry,
    McpToolInfo,
    McpToolWrapper,
)


# 模拟 MCP server 的 Python 脚本：stdin 读 NDJSON，stdout 写 NDJSON。
FAKE_MCP_SERVER = r"""
import json
import sys


def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _tool_echo(args):
    return args.get("text", "")


def _tool_add(args):
    return json.dumps({"sum": args.get("a", 0) + args.get("b", 0)})


TOOLS = {
    "echo": {
        "description": "回显输入文本",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "handler": _tool_echo,
    },
    "add": {
        "description": "加法",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        "handler": _tool_add,
    },
}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        _send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0.0"},
            },
        })
    elif method == "notifications/initialized":
        # 不需要响应
        pass
    elif method == "tools/list":
        _send({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": name,
                        "description": info["description"],
                        "inputSchema": info["inputSchema"],
                    }
                    for name, info in TOOLS.items()
                ]
            },
        })
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        info = TOOLS.get(name)
        if info is None:
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"unknown tool: {name}"},
            })
        else:
            try:
                result_text = info["handler"](args)
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                    },
                })
            except Exception as exc:
                _send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"error: {exc}"}],
                        "isError": True,
                    },
                })
"""


def _write_fake_server(temp_dir: str) -> str:
    path = Path(temp_dir) / "fake_mcp_server.py"
    path.write_text(FAKE_MCP_SERVER, encoding="utf-8")
    return str(path)


class McpClientConnectTest(unittest.IsolatedAsyncioTestCase):
    """client 启动 + initialize 握手。"""

    async def test_connect_lists_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(
                name="fake",
                command=[sys.executable, server_path],
            )
            try:
                await client.connect()
                self.assertTrue(client.is_connected)
                tool_names = sorted(t.name for t in client.tools)
                self.assertEqual(tool_names, ["add", "echo"])
            finally:
                await client.close()

    async def test_close_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="fake", command=[sys.executable, server_path])
            await client.connect()
            await client.close()
            self.assertFalse(client.is_connected)

    async def test_connect_with_invalid_command_raises(self) -> None:
        client = McpClient(
            name="missing",
            command=[sys.executable, "/nonexistent/__no_such__.py"],
        )
        with self.assertRaises(McpError):
            await client.connect()
        await client.close()


class McpCallToolTest(unittest.IsolatedAsyncioTestCase):
    """tools/call 调用远端工具。"""

    async def test_call_echo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="fake", command=[sys.executable, server_path])
            try:
                await client.connect()
                result = await client.call_tool("echo", {"text": "hello mcp"})
                self.assertEqual(result, "hello mcp")
            finally:
                await client.close()

    async def test_call_add(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="fake", command=[sys.executable, server_path])
            try:
                await client.connect()
                result = await client.call_tool("add", {"a": 2, "b": 3})
                data = json.loads(result)
                self.assertEqual(data["sum"], 5)
            finally:
                await client.close()

    async def test_call_unknown_tool_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="fake", command=[sys.executable, server_path])
            try:
                await client.connect()
                with self.assertRaises(McpError):
                    await client.call_tool("nope")
            finally:
                await client.close()

    async def test_call_without_connect_raises(self) -> None:
        client = McpClient(name="fake", command=[sys.executable, "-c", "pass"])
        with self.assertRaises(McpError):
            await client.call_tool("echo")


class McpToolWrapperTest(unittest.IsolatedAsyncioTestCase):
    """McpToolWrapper 把远端工具暴露成 Tool。"""

    async def test_wrapper_name_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="my_server", command=[sys.executable, server_path])
            try:
                await client.connect()
                infos = {t.name: t for t in client.tools}
                wrapper = McpToolWrapper(client, infos["echo"])
                self.assertEqual(wrapper.name, "mcp_my_server__echo")
                self.assertIn("[MCP:my_server]", wrapper.description)
                schema = wrapper.to_schema()
                self.assertEqual(schema["function"]["name"], "mcp_my_server__echo")
                self.assertEqual(schema["function"]["parameters"]["type"], "object")

                # execute 也能调用
                result = await wrapper.execute({"text": "wrapped"})
                self.assertEqual(result, "wrapped")
            finally:
                await client.close()


class McpRegistryTest(unittest.IsolatedAsyncioTestCase):
    """McpRegistry 聚合多个 client。"""

    async def test_collect_tools_from_multiple_servers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)

            registry = McpRegistry()
            registry.add("alpha", [sys.executable, server_path])
            registry.add("beta", [sys.executable, server_path])

            self.assertEqual(set(registry.get.__name__ if False else registry._clients.keys()),
                             {"alpha", "beta"})

            try:
                await registry.connect_all()
                tools = registry.collect_tools()
                # 每个 server 各暴露 echo + add
                self.assertEqual(len(tools), 4)
                names = sorted(t.name for t in tools)
                self.assertEqual(names, [
                    "mcp_alpha__add", "mcp_alpha__echo",
                    "mcp_beta__add", "mcp_beta__echo",
                ])
            finally:
                await registry.close_all()

    def test_add_duplicate_raises(self) -> None:
        registry = McpRegistry()
        registry.add("dup", [sys.executable, "-c", "pass"])
        with self.assertRaises(ValueError):
            registry.add("dup", [sys.executable, "-c", "pass"])

    def test_get_returns_none_for_unknown(self) -> None:
        registry = McpRegistry()
        self.assertIsNone(registry.get("nope"))


class SafeSplitCommandTest(unittest.TestCase):
    """_safe_split_command 命令字符串解析。"""

    def test_list_passes_through(self) -> None:
        from mind.mcp.client import _safe_split_command
        self.assertEqual(
            _safe_split_command(["python", "-c", "pass"]),
            ["python", "-c", "pass"],
        )

    def test_string_gets_split(self) -> None:
        from mind.mcp.client import _safe_split_command
        self.assertEqual(
            _safe_split_command("python -c pass"),
            ["python", "-c", "pass"],
        )

    def test_string_with_quotes(self) -> None:
        from mind.mcp.client import _safe_split_command
        # shlex.split 能正确处理引号
        self.assertEqual(
            _safe_split_command('python -c "hello world"'),
            ["python", "-c", "hello world"],
        )

    def test_empty_list_returns_empty(self) -> None:
        from mind.mcp.client import _safe_split_command
        self.assertEqual(_safe_split_command([]), [])


if __name__ == "__main__":
    unittest.main()