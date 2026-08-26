"""插件系统公共接口。"""

from extensions.manager import Extension, LoadedExtension, ExtensionManager, ExtensionMeta

__all__ = ["Extension", "LoadedExtension", "ExtensionManager", "ExtensionMeta"]
