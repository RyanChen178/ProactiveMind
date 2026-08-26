"""插件系统测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from mind.tools import ToolRegistry
from extensions.manager import Extension, ExtensionManager, ExtensionMeta


class FakeExtension(Extension):
    """测试用假扩展。"""

    meta = ExtensionMeta(name="fake", description="测试插件", version="0.0.1")
    loaded = False
    unloaded = False

    def register_tools(self, registry: ToolRegistry) -> None:
        async def _noop(_: dict) -> str:
            return "ok"

        from mind.tools import Tool

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


def _write_extension_file(extensions_dir: Path, name: str, content: str) -> None:
    extensions_dir.mkdir(parents=True, exist_ok=True)
    (extensions_dir / f"{name}.py").write_text(content, encoding="utf-8")


_VALID_EXTENSION = '''\
from extensions.manager import Extension, ExtensionMeta
from mind.tools import Tool, ToolRegistry

async def _hello(args):
    return "hello"

class HelloExtension(Extension):
    meta = ExtensionMeta(name="hello", description="hello 扩展", version="1.0.0")

    def register_tools(self, registry):
        registry.register(Tool(
            name="hello",
            description="说 hello",
            parameters={"type": "object", "properties": {}},
            func=_hello,
        ))

def create_extension():
    return HelloExtension()
'''

_BAD_EXTENSION = '''\
def create_extension():
    return "not an Extension"
'''

_NO_CREATE = '''\
print("no create_extension here")
'''


class ExtensionManagerTest(unittest.IsolatedAsyncioTestCase):
    def test_discover_finds_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            _write_extension_file(extensions_dir, "hello", _VALID_EXTENSION)
            _write_extension_file(extensions_dir, "_private", _VALID_EXTENSION)
            _write_extension_file(extensions_dir, "bad", _VALID_EXTENSION)

            mgr = ExtensionManager(extensions_dir)
            names = mgr.discover()

            self.assertEqual(names, ["bad", "hello"])

    def test_discover_returns_empty_when_no_dir(self) -> None:
        mgr = ExtensionManager(Path("/nonexistent/plugins"))
        self.assertEqual(mgr.discover(), [])

    def test_load_all_registers_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            _write_extension_file(extensions_dir, "hello", _VALID_EXTENSION)

            mgr = ExtensionManager(extensions_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].extension.meta.name, "hello")
            self.assertEqual(loaded[0].extension.meta.version, "1.0.0")
            self.assertIn("hello", loaded[0].tool_names)
            self.assertIn("hello", registry._tools)

    def test_load_all_skips_BAD_EXTENSION(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            _write_extension_file(extensions_dir, "valid", _VALID_EXTENSION)
            _write_extension_file(extensions_dir, "bad", _BAD_EXTENSION)

            mgr = ExtensionManager(extensions_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].extension.meta.name, "hello")

    def test_load_all_skips_module_without_create_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            _write_extension_file(extensions_dir, "noplugin", _NO_CREATE)
            _write_extension_file(extensions_dir, "hello", _VALID_EXTENSION)

            mgr = ExtensionManager(extensions_dir)
            registry = ToolRegistry()

            loaded = mgr.load_all(registry)

            self.assertEqual(len(loaded), 1)

    async def test_unload_all_calls_on_unload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "plugins"
            _write_extension_file(extensions_dir, "hello", _VALID_EXTENSION)

            mgr = ExtensionManager(extensions_dir)
            registry = ToolRegistry()
            mgr.load_all(registry)

            await mgr.unload_all()

            self.assertEqual(mgr.loaded, [])

    def test_loaded_property_returns_copy(self) -> None:
        mgr = ExtensionManager(Path("/nonexistent"))
        original = mgr.loaded
        original.append("fake")  # 不应影响内部列表
        self.assertEqual(mgr.loaded, [])


class ExtensionMetaTest(unittest.TestCase):
    def test_default_values(self) -> None:
        meta = ExtensionMeta(name="test")
        self.assertEqual(meta.name, "test")
        self.assertEqual(meta.description, "")
        self.assertEqual(meta.version, "0.0.1")
        self.assertEqual(meta.author, "")


if __name__ == "__main__":
    unittest.main()
