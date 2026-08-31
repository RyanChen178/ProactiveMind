"""扩展系统工具拦截器

支持在工具执行前拦截、修改参数、阻止执行或提供模拟结果。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

# 工具拦截器类型
ToolInterceptor = Callable[[str, dict[str, Any]], Any | Awaitable[Any]]


@dataclass
class ToolInterceptorResult:
    """工具拦截器返回结果"""
    should_block: bool = False
    block_reason: str = ""
    modified_arguments: dict[str, Any] | None = None
    mock_result: Any = None


class ToolInterceptorRegistry:
    """工具拦截器注册表"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._global_interceptors = []
            cls._instance._tool_interceptors = {}
        return cls._instance
    
    def register_global(self, interceptor: ToolInterceptor) -> None:
        """注册全局拦截器（应用于所有工具）"""
        self._global_interceptors.append(interceptor)
    
    def register_tool(self, tool_name: str, interceptor: ToolInterceptor) -> None:
        """注册特定工具的拦截器"""
        if tool_name not in self._tool_interceptors:
            self._tool_interceptors[tool_name] = []
        self._tool_interceptors[tool_name].append(interceptor)
    
    def unregister_global(self, interceptor: ToolInterceptor) -> None:
        """注销全局拦截器"""
        if interceptor in self._global_interceptors:
            self._global_interceptors.remove(interceptor)
    
    def unregister_tool(self, tool_name: str, interceptor: ToolInterceptor) -> None:
        """注销特定工具的拦截器"""
        if tool_name in self._tool_interceptors and interceptor in self._tool_interceptors[tool_name]:
            self._tool_interceptors[tool_name].remove(interceptor)
    
    def clear(self) -> None:
        """清空所有拦截器"""
        self._global_interceptors.clear()
        self._tool_interceptors.clear()
    
    async def intercept(self, tool_name: str, arguments: dict[str, Any]) -> ToolInterceptorResult:
        """执行拦截器链"""
        # 合并全局和工具特定的拦截器
        interceptors = list(self._global_interceptors)
        if tool_name in self._tool_interceptors:
            interceptors.extend(self._tool_interceptors[tool_name])
        
        current_arguments = arguments
        for interceptor in interceptors:
            try:
                result = interceptor(tool_name, current_arguments)
                if asyncio.iscoroutine(result):
                    result = await result
                
                if not isinstance(result, ToolInterceptorResult):
                    continue
                
                # 检查是否需要阻止执行
                if result.should_block:
                    return result
                
                # 更新参数（如果拦截器修改了参数）
                if result.modified_arguments is not None:
                    current_arguments = result.modified_arguments
                
            except Exception as e:
                # 拦截器异常不应该阻止执行
                import logging
                logging.getLogger(__name__).warning(
                    f"Error in interceptor for {tool_name}: {e}"
                )
        
        # 如果没有拦截器阻止，返回成功结果（可能包含修改后的参数）
        return ToolInterceptorResult(modified_arguments=current_arguments if current_arguments != arguments else None)


def get_tool_interceptor_registry() -> ToolInterceptorRegistry:
    """获取全局工具拦截器注册表"""
    return ToolInterceptorRegistry()


def register_global_interceptor(interceptor: ToolInterceptor) -> None:
    """注册全局拦截器的便捷函数"""
    get_tool_interceptor_registry().register_global(interceptor)


def register_tool_interceptor(tool_name: str, interceptor: ToolInterceptor) -> None:
    """注册特定工具拦截器的便捷函数"""
    get_tool_interceptor_registry().register_tool(tool_name, interceptor)


def unregister_global_interceptor(interceptor: ToolInterceptor) -> None:
    """注销全局拦截器的便捷函数"""
    get_tool_interceptor_registry().unregister_global(interceptor)


def unregister_tool_interceptor(tool_name: str, interceptor: ToolInterceptor) -> None:
    """注销特定工具拦截器的便捷函数"""
    get_tool_interceptor_registry().unregister_tool(tool_name, interceptor)


async def intercept_tool_call(tool_name: str, arguments: dict[str, Any]) -> ToolInterceptorResult:
    """拦截工具调用的便捷函数"""
    return await get_tool_interceptor_registry().intercept(tool_name, arguments)


def on_tool_pre(tool_name: str | None = None):
    """装饰器：标记函数为工具拦截器
    
    Usage:
        @on_tool_pre()  # 全局拦截器
        async def my_interceptor(tool_name, arguments):
            return ToolInterceptorResult()
        
        @on_tool_pre("shell")  # 特定工具拦截器
        async def shell_interceptor(tool_name, arguments):
            return ToolInterceptorResult()
    """
    def decorator(func: ToolInterceptor) -> ToolInterceptor:
        func._is_tool_interceptor = True
        if tool_name is None:
            register_global_interceptor(func)
        else:
            register_tool_interceptor(tool_name, func)
        return func
    return decorator


def is_tool_interceptor(func: Any) -> bool:
    """检查函数是否被标记为工具拦截器"""
    return getattr(func, '_is_tool_interceptor', False)


def get_intercept_tool(func: Any) -> str | None:
    """获取拦截器绑定的工具名称（None 表示全局）"""
    return getattr(func, '_intercept_tool', None)
