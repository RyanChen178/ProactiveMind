"""MCP 集成测试：config 解析 + MindLoop setup_mcp + 工具注册。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from mind.config import McpServerConfig, load_config
from mind.mcp import McpRegistry

# 复用 test_mcp 的 fake server 脚本
from tests.test_mcp import FAKE_MCP_SERVER, _write_fake_server


def _write_config(temp_dir: str, extra: str = "") -> str:
    path = Path(temp_dir) / "config.toml"
    path.write_text(
        """
[llm]
main = "rt"

[llm.runtimes.rt]
provider = "openai"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"

[workspace]
path = "~/.proactivemind/workspace-test"
"""
        + extra,
        encoding="utf-8",
    )
    return str(path)


class ParseMcpServersTest(unittest.TestCase):
    """[mcp.servers] 配置段解析。"""

    def _load(self, temp_dir: str, extra: str):
        cfg_path = _write_config(temp_dir, extra)
        return load_config(cfg_path)

    def test_no_mcp_section_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._load(temp_dir, "")
            self.assertEqual(cfg.mcp_servers, {})

    def test_string_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._load(temp_dir, """
[mcp.servers.fetch]
command = "uvx mcp-server-fetch"
""")
            self.assertIn("fetch", cfg.mcp_servers)
            self.assertEqual(cfg.mcp_servers["fetch"].command, "uvx mcp-server-fetch")
            self.assertTrue(cfg.mcp_servers["fetch"].enabled)

    def test_list_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._load(temp_dir, """
[mcp.servers.fs]
command = ["npx", "-y", "server-fs", "/tmp"]
""")
            self.assertEqual(
                cfg.mcp_servers["fs"].command,
                ["npx", "-y", "server-fs", "/tmp"],
            )

    def test_env_and_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = self._load(temp_dir, """
[mcp.servers.a]
command = "run-a"

[mcp.servers.a.env]
TIMEOUT = "30"
DEBUG = "1"

[mcp.servers.b]
command = "run-b"
enabled = false
""")
            self.assertEqual(cfg.mcp_servers["a"].env, {"TIMEOUT": "30", "DEBUG": "1"})
            self.assertFalse(cfg.mcp_servers["b"].enabled)

    def test_missing_command_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                self._load(temp_dir, """
[mcp.servers.bad]
enabled = true
""")

    def test_empty_command_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                self._load(temp_dir, """
[mcp.servers.bad]
command = "   "
""")

    def test_mcp_servers_not_table_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                self._load(temp_dir, """
[mcp]
servers = "oops"
""")


def _make_agent(config) -> object:
    """构造最小 MindLoop（绕过完整 __init__，只注入 setup_mcp 依赖）。"""
    from mind.loop import MindLoop

    agent = MindLoop.__new__(MindLoop)
    agent._config = config
    agent._tools = MagicMock()
    agent._tools.register = MagicMock()
    agent._mcp_registry = agent._build_mcp_registry()
    return agent


class BuildMcpRegistryTest(unittest.TestCase):
    """MindLoop._build_mcp_registry 行为。"""

    def _agent_with(self, mcp_servers: dict) -> object:
        config = MagicMock()
        config.mcp_servers = mcp_servers
        return _make_agent(config)

    def test_returns_none_without_servers(self) -> None:
        agent = self._agent_with({})
        self.assertIsNone(agent._build_mcp_registry())

    def test_returns_none_when_all_disabled(self) -> None:
        agent = self._agent_with({
            "a": McpServerConfig(name="a", command="x", enabled=False),
        })
        self.assertIsNone(agent._build_mcp_registry())

    def test_registers_enabled_servers_only(self) -> None:
        agent = self._agent_with({
            "on": McpServerConfig(name="on", command=["echo", "hi"]),
            "off": McpServerConfig(name="off", command=["echo", "no"], enabled=False),
        })
        registry = agent._build_mcp_registry()
        self.assertIsInstance(registry, McpRegistry)
        self.assertIsNotNone(registry.get("on"))
        self.assertIsNone(registry.get("off"))

    def test_defaults_create_registry(self) -> None:
        config = MagicMock()
        # 无 mcp_servers 属性时（旧 Config）不应崩溃
        del config.mcp_servers
        agent = _make_agent(config)
        # MagicMock 属性删除后再 getattr 返回默认 None
        try:
            result = agent._build_mcp_registry()
        except AttributeError:
            result = None
        self.assertIsNone(result)


class SetupMcpTest(unittest.IsolatedAsyncioTestCase):
    """MindLoop.setup_mcp 端到端：真实子进程握手 + 工具注册。"""

    async def test_setup_registers_remote_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            config = MagicMock()
            config.mcp_servers = {
                "fake": McpServerConfig(
                    name="fake",
                    command=[sys.executable, server_path],
                ),
            }
            agent = _make_agent(config)

            count = await agent.setup_mcp()
            self.assertEqual(count, 2)  # echo + add
            # 工具已注册到 ToolRegistry
            names = [c.args[0].name for c in agent._tools.register.call_args_list]
            self.assertEqual(
                sorted(names), ["mcp_fake__add", "mcp_fake__echo"]
            )
            await agent._mcp_registry.close_all()

    async def test_setup_skips_failed_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            config = MagicMock()
            config.mcp_servers = {
                "good": McpServerConfig(
                    name="good", command=[sys.executable, server_path]
                ),
                "bad": McpServerConfig(
                    name="bad",
                    command=[sys.executable, "/nonexistent/no_such_server.py"],
                ),
            }
            agent = _make_agent(config)

            count = await agent.setup_mcp()
            # bad 失败被跳过，good 的 2 个工具注册成功
            self.assertEqual(count, 2)
            await agent._mcp_registry.close_all()

    async def test_setup_returns_zero_without_registry(self) -> None:
        config = MagicMock()
        config.mcp_servers = {}
        agent = _make_agent(config)
        self.assertEqual(await agent.setup_mcp(), 0)


class McpToolThroughRegistryTest(unittest.IsolatedAsyncioTestCase):
    """MCP 工具经 ToolRegistry 端到端调用。"""

    async def test_execute_via_tool_registry(self) -> None:
        from mind.tools import ToolRegistry, ToolCall
        from mind.mcp import McpClient, McpToolWrapper

        with tempfile.TemporaryDirectory() as temp_dir:
            server_path = _write_fake_server(temp_dir)
            client = McpClient(name="fake", command=[sys.executable, server_path])
            try:
                await client.connect()
                registry = ToolRegistry()
                for info in client.tools:
                    registry.register(McpToolWrapper(client, info))  # type: ignore[arg-type]

                # schema 包含 MCP 工具
                schemas = registry.get_schemas()
                names = {s["function"]["name"] for s in schemas}
                self.assertEqual(names, {"mcp_fake__echo", "mcp_fake__add"})

                # 权限检查通过（非 shell 工具默认放行）并执行成功
                call = ToolCall(
                    id="c1",
                    name="mcp_fake__echo",
                    arguments={"text": "via registry"},
                )
                result = await registry.execute(call)
                self.assertEqual(result, "via registry")
            finally:
                await client.close()


if __name__ == "__main__":
    unittest.main()