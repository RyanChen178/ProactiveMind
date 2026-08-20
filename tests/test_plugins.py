"""插件系统测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.tools import ToolRegistry
from plugins.manager import Plugin, PluginManager, PluginMeta


class FakePlugin(Plugin):
    """测试用假插件。"""

    meta = PluginMeta(name="fake", description="测试插件", version="0.0.1")
    loaded = False
    unloaded = False

    def register_tools(self, registry: ToolRegistry) -> None:
        async def _noop(_: dict) -> str:
            return "ok"

        from agent.tools import Tool

        registry.register(
            Tool(
                name="fake_tool",
                description="假工具",
                parameters={"type": "object", "properties": {}},
                func=_noop,
            )
        )

    async def on_load(self) -> None:
        self.loaded = True

    async def on_unload(self) -> None:
        self.unloaded = True


def _write_plugin_file(plugins_dir: Path, name: str, content: str) -> None:
    plugins_dir.mkdir(parents=True, exist_ok=True)
    (plugins_dir / f"{name}.py").write_text(content, encoding="utf-8")


_VALID_PLUGIN = '''\
from plugins.manager import Plugin, PluginMeta
from agent.tools import Tool, ToolRegistry

async def _hello(args):
    return "hello"

class HelloPlugin(Plugin):
    meta = PluginMeta(name="hello", description="hello 插件", version="1.0.0")

    def register_tools(self, registry):
        registry.register(Tool(
            name="hello",
            description="说 hello",
            parameters={"type": "object", "properties": {}},
            func=_hello,
        ))

def create_plugin():
    return HelloPlugin()
'''

_BAD_PLUGIN = '''\
def create_plugin():
    return "not a plugin"
'''

_NO_CREATE = '''\
print("no create_plugin here")
'''


class PluginManagerTest(unittest.IsolatedAsyncioTestCase):
    def test_discover_finds_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            _write_plugin_file(plugins_dir, "hello", _VALID_PLUGIN)
            _write_plugin_file(plugins_dir, "_private", _VALID_PLUGIN)
            _write_plugin_file(plugins_dir, "bad", _VALID_PLUGIN)

            mgr = PluginManager(plugins_dir)
            names = mgr.discover()

            self.assertEqual(names, ["bad", "hello"])

    def test_discover_returns_empty_when_no_dir(self) -> None:
        mgr = PluginManager(Path("/nonexistent/plugins"))
        self.assertEqual(mgr.discover(), [])

    def test_load_all_registers_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            _write_plugin_file(plugins_dir, "hello", _VALID_PLUGIN)

            mgr = PluginManager(plugins_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].plugin.meta.name, "hello")
            self.assertEqual(loaded[0].plugin.meta.version, "1.0.0")
            self.assertIn("hello", loaded[0].tool_names)
            self.assertIn("hello", registry._tools)

    def test_load_all_skips_bad_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            _write_plugin_file(plugins_dir, "valid", _VALID_PLUGIN)
            _write_plugin_file(plugins_dir, "bad", _BAD_PLUGIN)

            mgr = PluginManager(plugins_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].plugin.meta.name, "hello")

    def test_load_all_skips_module_without_create_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            _write_plugin_file(plugins_dir, "noplugin", _NO_CREATE)
            _write_plugin_file(plugins_dir, "hello", _VALID_PLUGIN)

            mgr = PluginManager(plugins_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)

    async def test_unload_all_calls_on_unload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plugins_dir = Path(temp_dir) / "plugins"
            _write_plugin_file(plugins_dir, "hello", _VALID_PLUGIN)

            mgr = PluginManager(plugins_dir)
            registry = ToolRegistry()
            mgr.load_all(registry)

            await mgr.unload_all()

            self.assertEqual(mgr.loaded, [])

    def test_loaded_property_returns_copy(self) -> None:
        mgr = PluginManager(Path("/nonexistent"))
        original = mgr.loaded
        original.append("fake")  # 不应影响内部列表
        self.assertEqual(mgr.loaded, [])


class PluginMetaTest(unittest.TestCase):
    def test_default_values(self) -> None:
        meta = PluginMeta(name="test")
        self.assertEqual(meta.name, "test")
        self.assertEqual(meta.description, "")
        self.assertEqual(meta.version, "0.0.1")
        self.assertEqual(meta.author, "")


if __name__ == "__main__":
    unittest.main()
