"""插件系统 —— 声明式工具注册 + 自动发现。

每个插件是一个 Python 模块，放在 plugins/ 目录下，
模块中定义 create_plugin() 返回 Plugin 实例。

PluginManager 启动时扫描目录，加载所有插件，注册工具。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.tools import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class PluginMeta:
    """插件元信息。"""

    name: str
    description: str = ""
    version: str = "0.0.1"
    author: str = ""


class Plugin:
    """插件基类——子类通过 register_tools 注册工具。"""

    meta: PluginMeta = PluginMeta(name="unnamed")

    def register_tools(self, registry: ToolRegistry) -> None:
        """把本插件的工具注册到全局 ToolRegistry。"""
        pass

    async def on_load(self) -> None:
        """插件加载时调用（可选）。"""
        pass

    async def on_unload(self) -> None:
        """插件卸载时调用（可选）。"""
        pass


@dataclass
class LoadedPlugin:
    """已加载的插件实例。"""

    plugin: Plugin
    module_path: str
    tool_names: list[str] = field(default_factory=list)


class PluginManager:
    """插件管理器——发现、加载、注册。"""

    def __init__(self, plugins_dir: Path) -> None:
        self._plugins_dir = plugins_dir
        self._loaded: list[LoadedPlugin] = []

    def discover(self) -> list[str]:
        """扫描 plugins 目录，返回可导入的模块名列表。"""
        if not self._plugins_dir.exists():
            return []

        modules: list[str] = []
        for child in sorted(self._plugins_dir.iterdir()):
            if child.suffix != ".py":
                continue
            if child.name.startswith("_"):
                continue
            modules.append(child.stem)
        return modules

    def load_all(self, registry: ToolRegistry) -> list[LoadedPlugin]:
        """发现并加载所有插件，注册工具到 registry。"""
        module_names = self.discover()
        results: list[LoadedPlugin] = []

        for mod_name in module_names:
            try:
                plugin = self._load_module(mod_name)
                if plugin is None:
                    continue

                before = set(registry._tools.keys())
                plugin.register_tools(registry)
                after = set(registry._tools.keys())
                new_tools = sorted(after - before)

                loaded = LoadedPlugin(
                    plugin=plugin,
                    module_path=f"{self._plugins_dir.name}/{mod_name}.py",
                    tool_names=new_tools,
                )
                self._loaded.append(loaded)
                results.append(loaded)
                log.info(
                    "插件 %s v%s 已加载，注册工具: %s",
                    plugin.meta.name,
                    plugin.meta.version,
                    new_tools or "(无)",
                )
            except Exception as exc:
                log.warning("加载插件 %s 失败: %s", mod_name, exc)

        return results

    def _load_module(self, mod_name: str) -> Plugin | None:
        """从文件加载插件模块并调用 create_plugin()。"""
        file_path = self._plugins_dir / f"{mod_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"_plugin_{mod_name}", file_path
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_plugin_{mod_name}"] = module
        spec.loader.exec_module(module)

        create_fn = getattr(module, "create_plugin", None)
        if create_fn is None:
            log.warning("插件 %s 缺少 create_plugin() 函数", mod_name)
            return None

        plugin = create_fn()
        if not isinstance(plugin, Plugin):
            log.warning("插件 %s 的 create_plugin() 未返回 Plugin 实例", mod_name)
            return None

        return plugin

    @property
    def loaded(self) -> list[LoadedPlugin]:
        """已加载的插件列表。"""
        return list(self._loaded)

    async def unload_all(self) -> None:
        """卸载所有插件。"""
        for loaded in reversed(self._loaded):
            try:
                await loaded.plugin.on_unload()
            except Exception as exc:
                log.warning("卸载插件 %s 失败: %s", loaded.plugin.meta.name, exc)
        self._loaded.clear()
