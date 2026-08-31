"""扩展系统热重载

支持在不重启进程的情况下重新加载扩展。
使用代际（generation）机制来管理扩展版本。
"""

from __future__ import annotations

import os
import time
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtensionFileState:
    """扩展文件状态"""
    path: Path
    mtime: float
    size: int
    generation: int


@dataclass
class GenerationInfo:
    """代际信息"""
    generation: int
    timestamp: float
    extensions: list[str]


class HotReloader:
    """扩展热重载器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._extensions_dir = None
            cls._instance._file_states = {}
            cls._instance._current_generation = 0
            cls._instance._generations = {}
            cls._instance._running = False
            cls._instance._poll_interval = 2.0
            cls._instance._on_reload = None
            cls._instance._monitor_task = None
        return cls._instance
    
    def init(self, extensions_dir: Path, poll_interval: float = 2.0, on_reload: Any = None) -> None:
        """初始化热重载器"""
        self._extensions_dir = extensions_dir
        self._poll_interval = poll_interval
        self._on_reload = on_reload
        self._current_generation = 0
        self._generations[0] = GenerationInfo(
            generation=0,
            timestamp=time.time(),
            extensions=[]
        )
    
    @property
    def current_generation(self) -> int:
        """获取当前代际"""
        return self._current_generation
    
    def get_generation_info(self, generation: int) -> GenerationInfo | None:
        """获取指定代际的信息"""
        return self._generations.get(generation)
    
    def acquire_generation(self) -> int:
        """获取当前代际（用于新请求）"""
        return self._current_generation
    
    def release_generation(self, generation: int) -> None:
        """释放代际（请求完成后调用）"""
        # 简单实现：不立即清理旧代际，等待下次重载时清理
        pass
    
    def _scan_files(self) -> dict[str, ExtensionFileState]:
        """扫描扩展目录中的所有文件"""
        if not self._extensions_dir or not self._extensions_dir.exists():
            return {}
        
        states = {}
        for file_path in self._extensions_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue
            
            stat = file_path.stat()
            module_name = file_path.stem
            states[module_name] = ExtensionFileState(
                path=file_path,
                mtime=stat.st_mtime,
                size=stat.st_size,
                generation=self._current_generation
            )
        
        return states
    
    def _detect_changes(self, new_states: dict[str, ExtensionFileState]) -> tuple[list[str], list[str], list[str]]:
        """检测文件变化"""
        old_states = self._file_states
        
        added = []
        modified = []
        deleted = []
        
        # 检查新增和修改
        for module_name, new_state in new_states.items():
            if module_name not in old_states:
                added.append(module_name)
            else:
                old_state = old_states[module_name]
                if new_state.mtime != old_state.mtime or new_state.size != old_state.size:
                    modified.append(module_name)
        
        # 检查删除
        for module_name in old_states:
            if module_name not in new_states:
                deleted.append(module_name)
        
        return added, modified, deleted
    
    async def check_and_reload(self) -> bool:
        """检查并重新加载扩展"""
        if not self._extensions_dir:
            return False
        
        new_states = self._scan_files()
        added, modified, deleted = self._detect_changes(new_states)
        
        if not (added or modified or deleted):
            return False
        
        logger.info(f"Detected changes: added={added}, modified={modified}, deleted={deleted}")
        
        # 更新文件状态
        self._file_states = new_states
        
        # 增加代际
        self._current_generation += 1
        self._generations[self._current_generation] = GenerationInfo(
            generation=self._current_generation,
            timestamp=time.time(),
            extensions=list(new_states.keys())
        )
        
        # 调用重载回调
        if self._on_reload:
            try:
                result = self._on_reload(added, modified, deleted)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")
        
        # 重新加载修改和新增的扩展
        for module_name in added + modified:
            await self._reload_extension(module_name)
        
        return True
    
    async def _reload_extension(self, module_name: str) -> None:
        """重新加载单个扩展"""
        if module_name not in self._file_states:
            return
        
        file_state = self._file_states[module_name]
        module_full_name = f"extensions.{module_name}"
        
        try:
            # 如果模块已加载，先清理
            if module_full_name in importlib.sys.modules:
                old_module = importlib.sys.modules[module_full_name]
                # 调用清理函数（如果存在）
                if hasattr(old_module, 'cleanup'):
                    cleanup_result = old_module.cleanup()
                    if asyncio.iscoroutine(cleanup_result):
                        await cleanup_result
                
                # 删除旧模块
                del importlib.sys.modules[module_full_name]
            
            # 重新加载模块
            spec = importlib.util.spec_from_file_location(
                module_full_name,
                file_state.path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                importlib.sys.modules[module_full_name] = module
                spec.loader.exec_module(module)
                
                logger.info(f"Reloaded extension: {module_name} (generation {self._current_generation})")
        
        except Exception as e:
            logger.error(f"Failed to reload extension {module_name}: {e}")
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            await self.check_and_reload()
            await asyncio.sleep(self._poll_interval)
    
    def start(self) -> None:
        """启动热重载监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started hot reload monitor (poll_interval={self._poll_interval}s)")
    
    def stop(self) -> None:
        """停止热重载监控"""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("Stopped hot reload monitor")


def get_hot_reloader() -> HotReloader:
    """获取全局热重载器"""
    return HotReloader()


def init_hot_reloader(extensions_dir: Path, poll_interval: float = 2.0, on_reload: Any = None) -> HotReloader:
    """初始化热重载器"""
    reloader = get_hot_reloader()
    reloader.init(extensions_dir, poll_interval, on_reload)
    return reloader


def stop_hot_reloader() -> None:
    """停止热重载器"""
    reloader = get_hot_reloader()
    reloader.stop()
