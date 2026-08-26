"""插件集成到 MindLoop 的测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mind.loop import MindLoop
from mind.tools import ToolRegistry


_EXTENSION_CODE = '''\
from extensions.manager import Extension, ExtensionMeta
from mind.tools import Tool, ToolRegistry

async def _echo(args):
    return args.get("text", "")

class EchoExtension(Extension):
    meta = ExtensionMeta(name="echo", version="0.1.0")

    def register_tools(self, registry):
        registry.register(Tool(
            name="echo",
            description="回显输入",
            parameters={"type": "object", "properties": {
                "text": {"type": "string"}
            }, "required": ["text"]},
            func=_echo,
        ))

def create_extension():
    return EchoExtension()
'''


class PluginIntegrationTest(unittest.TestCase):
    def test_agent_loop_loads_plugins_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            extensions_dir.mkdir()
            (extensions_dir / "echo.py").write_text(_EXTENSION_CODE, encoding="utf-8")

            captured_tools: ToolRegistry = {}

            original_init = MindLoop.__init__

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
                self._extension_manager = None
                self._load_extensions()
                captured_tools["registry"] = self._tools

            with patch.object(MindLoop, "__init__", patched_init):
                config = SimpleNamespace(
                    extensions_dir=extensions_dir,
                )
                MindLoop(config)

            self.assertIn("echo", captured_tools["registry"]._tools)

    def test_agent_loop_skips_plugins_when_dir_none(self) -> None:
        captured = {}

        original_init = MindLoop.__init__

        def patched_init(self, config, **kwargs):
            self._config = config
            self._tools = ToolRegistry()
            self._extension_manager = None
            self._load_extensions()
            captured["pm"] = self._extension_manager

        with patch.object(MindLoop, "__init__", patched_init):
            config = SimpleNamespace(extensions_dir=None)
            MindLoop(config)

        self.assertIsNone(captured["pm"])

    def test_agent_loop_skips_plugins_when_dir_missing(self) -> None:
        captured = {}

        original_init = MindLoop.__init__

        def patched_init(self, config, **kwargs):
            self._config = config
            self._tools = ToolRegistry()
            self._extension_manager = None
            self._load_extensions()
            captured["pm"] = self._extension_manager

        with patch.object(MindLoop, "__init__", patched_init):
            config = SimpleNamespace(
                extensions_dir=Path("/nonexistent/plugins"),
            )
            MindLoop(config)

        self.assertIsNone(captured["pm"])


if __name__ == "__main__":
    unittest.main()
