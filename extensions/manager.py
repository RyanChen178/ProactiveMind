"""扩展系统 —— 声明式工具注册 + 自动发现。

每个扩展是一个 Python 模块，放在 extensions/ 目录下，
模块中定义 create_extension() 返回 Extension 实例。

ExtensionManager 启动时扫描目录，加载所有扩展，注册工具。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mind.tools import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class ExtensionMeta:
    """扩展元信息。"""

    name: str
    description: str = ""
    version: str = "0.0.1"
    author: str = ""


class Extension:
    """扩展基类——子类通过 register_tools 注册工具。"""

    meta: ExtensionMeta = ExtensionMeta(name="unnamed")

    def register_tools(self, registry: ToolRegistry) -> None:
        """把本扩展的工具注册到全局 ToolRegistry。"""
        pass

    async def on_load(self) -> None:
        """扩展加载时调用（可选）。"""
        pass

    async def on_unload(self) -> None:
        """扩展卸载时调用（可选）。"""
        pass


@dataclass
class LoadedExtension:
    """已加载的扩展实例。"""

    extension: Extension
    module_path: str
    tool_names: list[str] = field(default_factory=list)


class ExtensionManager:
    """扩展管理器——发现、加载、注册。"""

    def __init__(self, extensions_dir: Path) -> None:
        self._extensions_dir = extensions_dir
        self._loaded: list[LoadedExtension] = []

    def discover(self) -> list[str]:
        """扫描 extensions 目录，返回可导入的模块名列表。"""
        if not self._extensions_dir.exists():
            return []

        modules: list[str] = []
        for child in sorted(self._extensions_dir.iterdir()):
            if child.suffix != ".py":
                continue
            if child.name.startswith("_"):
                continue
            modules.append(child.stem)
        return modules

    def load_all(self, registry: ToolRegistry) -> list[LoadedExtension]:
        """发现并加载所有扩展，注册工具到 registry。"""
        module_names = self.discover()
        results: list[LoadedExtension] = []

        for mod_name in module_names:
            try:
                ext = self._load_module(mod_name)
                if ext is None:
                    continue

                before = set(registry._tools.keys())
                ext.register_tools(registry)
                after = set(registry._tools.keys())
                new_tools = sorted(after - before)

                loaded = LoadedExtension(
                    extension=ext,
                    module_path=f"{self._extensions_dir.name}/{mod_name}.py",
                    tool_names=new_tools,
                )
                self._loaded.append(loaded)
                results.append(loaded)
                log.info(
                    "扩展 %s v%s 已加载，注册工具: %s",
                    ext.meta.name,
                    ext.meta.version,
                    new_tools or "(无)",
                )
            except Exception as exc:
                log.warning("加载扩展 %s 失败: %s", mod_name, exc)

        return results

    def _load_module(self, mod_name: str) -> Extension | None:
        """从文件加载扩展模块并调用 create_extension()。"""
        file_path = self._extensions_dir / f"{mod_name}.py"
        spec = importlib.util.spec_from_file_location(
            f"_extension_{mod_name}", file_path
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_extension_{mod_name}"] = module
        spec.loader.exec_module(module)

        create_fn = getattr(module, "create_extension", None)
        if create_fn is None:
            log.warning("扩展 %s 缺少 create_extension() 函数", mod_name)
            return None

        ext = create_fn()
        if not isinstance(ext, Extension):
            log.warning("扩展 %s 的 create_extension() 未返回 Extension 实例", mod_name)
            return None

        return ext

    @property
    def loaded(self) -> list[LoadedExtension]:
        """已加载的扩展列表。"""
        return list(self._loaded)

    async def unload_all(self) -> None:
        """卸载所有扩展。"""
        for loaded in reversed(self._loaded):
            try:
                await loaded.extension.on_unload()
            except Exception as exc:
                log.warning("卸载扩展 %s 失败: %s", loaded.extension.meta.name, exc)
        self._loaded.clear()
