"""MCP Client —— Model Context Protocol 客户端实现。

对齐 MCP（Model Context Protocol）规范：
  - JSON-RPC 2.0 over stdio
  - 启动 server 子进程，通过 stdin/stdout 收发消息
  - 标准方法：initialize / initialized / tools/list / tools/call
  - 工具命名：mcp_<server>__<tool>

设计：
  - McpClient 负责单个 server 的生命周期
  - McpRegistry 聚合多个 client，把它们暴露的工具注册到 ToolRegistry
  - asyncio.subprocess + StreamReader/Writer 收发 NDJSON
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-11-25"
CLIENT_NAME = "proactivemind"
CLIENT_VERSION = "0.1.0"

INITIALIZE_TIMEOUT_S = 8.0
REQUEST_TIMEOUT_S = 30.0
RECV_BUFFER_LIMIT = 4 * 1024 * 1024  # 4 MB


class McpError(RuntimeError):
    """MCP 协议错误。"""

    pass


@dataclass
class McpToolInfo:
    """远端 MCP server 暴露的工具描述。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


class _JsonRpcState:
    """JSON-RPC 客户端：管理 request id 与 pending future。"""

    def __init__(self) -> None:
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._server_request_handler: Any = None

    def next_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def register(self, request_id: int) -> asyncio.Future[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        return future

    def resolve(self, msg: dict[str, Any]) -> bool:
        """收到响应或错误时调用；返回是否已处理。"""
        request_id = msg.get("id")
        if not isinstance(request_id, int):
            return False
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return True
        if "error" in msg:
            err = msg["error"]
            future.set_exception(McpError(
                f"MCP error {err.get('code', '?')}: {err.get('message', '?')}"
            ))
        else:
            future.set_result(msg)
        return True

    @property
    def pending(self) -> dict[int, asyncio.Future[dict[str, Any]]]:
        return self._pending


def _safe_split_command(command: str | list[str]) -> list[str]:
    """解析命令行字符串为参数列表（兼容 str 和 list 输入）。"""
    if isinstance(command, list):
        return [str(a) for a in command]
    return shlex.split(command)


class McpClient:
    """单个 MCP server 客户端：stdin/stdout JSON-RPC。"""

    def __init__(
        self,
        name: str,
        command: str | list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self._command = _safe_split_command(command)
        if not self._command:
            raise ValueError("command 不能为空")
        self._env = env
        self._cwd = cwd
        self._proc: asyncio.subprocess.Process | None = None
        self._rpc = _JsonRpcState()
        self._reader_task: asyncio.Task[None] | None = None
        self._tools: list[McpToolInfo] = []
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    async def connect(self) -> None:
        """启动 server 子进程并完成 initialize 握手。"""
        if self._connected:
            return
        env = os.environ.copy()
        if self._env:
            env.update(self._env)
        # Windows 下 CREATE_NEW_PROCESS_GROUP 便于发 CTRL_BREAK_EVENT
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess_CREATE_NEW_PROCESS_GROUP  # type: ignore[name-defined]
            )
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._cwd,
                limit=RECV_BUFFER_LIMIT,
                **kwargs,
            )
        except FileNotFoundError as exc:
            raise McpError(f"无法启动 MCP server {self.name!r}: {exc}") from exc

        assert self._proc.stdout is not None
        assert self._proc.stdin is not None
        self._reader_task = asyncio.create_task(
            self._reader_loop(self._proc.stdout, self._proc.stderr),
            name=f"mcp-reader-{self.name}",
        )

        try:
            await asyncio.wait_for(self._initialize(), timeout=INITIALIZE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise McpError(
                f"MCP server {self.name!r} initialize 超时"
            ) from exc
        self._connected = True

    async def _initialize(self) -> None:
        """发送 initialize + initialized 通知，再列出工具。"""
        request_id = self._rpc.next_id()
        self._rpc.register(request_id)
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "version": CLIENT_VERSION,
                },
            },
        })
        # 等响应
        future = self._rpc.pending[request_id]
        # _reader_loop 在后台 resolve；这里直接 await
        try:
            await future
        except McpError:
            raise

        # 发送 initialized 通知（无 id，不等响应）
        await self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

        # 列出工具
        list_id = self._rpc.next_id()
        self._rpc.register(list_id)
        await self._send({
            "jsonrpc": "2.0",
            "id": list_id,
            "method": "tools/list",
            "params": {},
        })
        try:
            response = await self._rpc.pending[list_id]
        except McpError as exc:
            raise McpError(f"MCP tools/list 失败: {exc}") from exc
        tools_raw = response.get("result", {}).get("tools") or []
        if not isinstance(tools_raw, list):
            raise McpError(f"MCP tools/list 返回非 list: {type(tools_raw).__name__}")

        self._tools = []
        for item in tools_raw:
            if not isinstance(item, dict):
                continue
            self._tools.append(McpToolInfo(
                name=str(item.get("name", "")),
                description=str(item.get("description", "")),
                input_schema=item.get("inputSchema") or {},
            ))

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpError("MCP client 未连接")
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(encoded)
            await self._proc.stdin.drain()
        except (ConnectionError, BrokenPipeError) as exc:
            raise McpError(f"MCP 写入失败: {exc}") from exc

    async def _reader_loop(
        self,
        stdout: asyncio.StreamReader,
        stderr: asyncio.StreamReader,
    ) -> None:
        """单 reader 分发响应、通知；stderr 单独打日志。"""
        async def _stderr_pump() -> None:
            while True:
                line = await stderr.readline()
                if not line:
                    return
                try:
                    text = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if text:
                    log.warning("[mcp:%s] stderr: %s", self.name, text)

        stderr_task = asyncio.create_task(_stderr_pump())
        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("[mcp:%s] 解析 JSON 失败: %s", self.name, exc)
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg:
                    self._rpc.resolve(msg)
                # 通知类无 id，本实现不主动处理 server -> client 请求
        except Exception as exc:
            log.warning("[mcp:%s] reader 异常: %s", self.name, exc)
        finally:
            # 关闭所有 pending future
            for future in list(self._rpc.pending.values()):
                if not future.done():
                    future.set_exception(McpError("MCP 连接已关闭"))
            self._rpc.pending.clear()
            stderr_task.cancel()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> str:
        """调用 MCP 远端工具，返回文本内容。"""
        if not self._connected:
            raise McpError("MCP client 未连接")

        request_id = self._rpc.next_id()
        self._rpc.register(request_id)
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        })
        try:
            response = await asyncio.wait_for(
                self._rpc.pending[request_id], timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise McpError(f"MCP tools/call {name!r} 超时") from exc
        except McpError:
            raise

        result = response.get("result") or {}
        content = result.get("content") if isinstance(result, dict) else None
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        is_error = bool(result.get("isError", False))
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str):
                chunks.append(text)
        joined = "".join(chunks)
        if is_error:
            raise McpError(f"MCP tool {name!r} 返回错误: {joined}")
        return joined

    async def close(self) -> None:
        """关闭 MCP 连接并清理资源。"""
        self._connected = False
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
        finally:
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._proc = None


class McpToolWrapper:
    """把 MCP 远端工具包装成本地 Tool 接口。"""

    def __init__(self, client: McpClient, info: McpToolInfo) -> None:
        self._client = client
        self._info = info

    @property
    def name(self) -> str:
        return f"mcp_{self._client.name}__{self._info.name}"

    @property
    def description(self) -> str:
        return f"[MCP:{self._client.name}] {self._info.description}"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._info.input_schema or {
            "type": "object",
            "properties": {},
        }

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return await self._client.call_tool(self._info.name, arguments or {})
        except McpError as exc:
            return f"[MCP 错误] {exc}"


class McpRegistry:
    """聚合多个 MCP server 的客户端管理。"""

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}

    def add(self, name: str, command: str | list[str], **kwargs: Any) -> McpClient:
        """注册一个新 client（未启动）。"""
        if name in self._clients:
            raise ValueError(f"MCP server {name!r} 已注册")
        client = McpClient(name=name, command=command, **kwargs)
        self._clients[name] = client
        return client

    def get(self, name: str) -> McpClient | None:
        return self._clients.get(name)

    async def connect_all(self) -> None:
        """启动所有 client 并完成握手。"""
        for client in self._clients.values():
            if not client.is_connected:
                await client.connect()

    async def close_all(self) -> None:
        """关闭所有 client。"""
        for client in self._clients.values():
            await client.close()

    def collect_tools(self) -> list[McpToolWrapper]:
        """收集所有已连接 client 的远端工具。"""
        wrappers: list[McpToolWrapper] = []
        for client in self._clients.values():
            if not client.is_connected:
                continue
            for info in client.tools:
                wrappers.append(McpToolWrapper(client, info))
        return wrappers


# Windows 上 subprocess 的 CREATE_NEW_PROCESS_GROUP 常量补全
if sys.platform == "win32":
    try:
        subprocess_CREATE_NEW_PROCESS_GROUP = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    except AttributeError:
        subprocess_CREATE_NEW_PROCESS_GROUP = 0  # type: ignore[assignment]
else:
    subprocess_CREATE_NEW_PROCESS_GROUP = 0  # type: ignore[assignment]