"""插件集成到 AgentLoop 的测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import AgentLoop
from agent.tools import ToolRegistry


_PLUGIN_CODE = '''\
from plugins.manager import Plugin, PluginMeta
from agent.tools import Tool, ToolRegistry

async def _echo(args):
    return args.get("text", "")

class EchoPlugin(Plugin):
    meta = PluginMeta(name="echo", version="0.1.0")

    def register_tools(self, registry):
        registry.register(Tool(
            name="echo",
            description="回显输入",
            parameters={"type": "object", "properties": {
                "text": {"type": "string"}
            }, "required": ["text"]},
            func=_echo,
        ))

def create_plugin():
    return EchoPlugin()
'''


class PluginIntegrationTest(unittest.TestCase):
    def test_agent_loop_loads_plugins_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            plugins_dir.mkdir()
            (plugins_dir / "echo.py").write_text(_PLUGIN_CODE, encoding="utf-8")

            captured_tools: ToolRegistry = {}

            original_init = AgentLoop.__init__

            def patched_init(self, config, **kwargs):
                self._config = config
                self._provider = None
                self._memory = None
                self._session_store = None
                self._session_id = "test"
                self._session = None
                self._tools = ToolRegistry()
                self._consolidator = None
                self._bus = kwargs.get("bus") or None
                self._presence = kwargs.get("presence")
                self._plugin_manager = None
                self._load_plugins()
                captured_tools["registry"] = self._tools

            with patch.object(AgentLoop, "__init__", patched_init):
                config = SimpleNamespace(
                    plugins_dir=plugins_dir,
                )
                AgentLoop(config)

            self.assertIn("echo", captured_tools["registry"]._tools)

    def test_agent_loop_skips_plugins_when_dir_none(self) -> None:
        captured = {}

        original_init = AgentLoop.__init__

        def patched_init(self, config, **kwargs):
            self._config = config
            self._tools = ToolRegistry()
            self._plugin_manager = None
            self._load_plugins()
            captured["pm"] = self._plugin_manager

        with patch.object(AgentLoop, "__init__", patched_init):
            config = SimpleNamespace(plugins_dir=None)
            AgentLoop(config)

        self.assertIsNone(captured["pm"])

    def test_agent_loop_skips_plugins_when_dir_missing(self) -> None:
        captured = {}

        original_init = AgentLoop.__init__

        def patched_init(self, config, **kwargs):
            self._config = config
            self._tools = ToolRegistry()
            self._plugin_manager = None
            self._load_plugins()
            captured["pm"] = self._plugin_manager

        with patch.object(AgentLoop, "__init__", patched_init):
            config = SimpleNamespace(
                plugins_dir=Path("/nonexistent/plugins"),
            )
            AgentLoop(config)

        self.assertIsNone(captured["pm"])


if __name__ == "__main__":
    unittest.main()
