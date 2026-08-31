"""扩展系统生命周期钩子

支持 6 阶段生命周期：
- before_turn: 在 turn 开始之前
- before_reasoning: 在推理开始之前
- prompt_render: 在 prompt 渲染时
- reasoner: 推理过程中
- after_reasoning: 推理完成之后
- after_turn: 在 turn 完成之后
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

# 生命周期钩子类型
LifecycleHook = Callable[[Any], Any | Awaitable[Any]]


@dataclass
class TurnContext:
    """Turn 执行上下文"""
    session_id: str
    user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    should_skip: bool = False
    skip_reason: str = ""


class LifecycleHooks:
    """管理扩展系统的生命周期钩子"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._hooks = {
                "before_turn": [],
                "before_reasoning": [],
                "prompt_render": [],
                "reasoner": [],
                "after_reasoning": [],
                "after_turn": [],
            }
        return cls._instance
    
    def register(self, phase: str, hook: LifecycleHook) -> None:
        """注册生命周期钩子"""
        if phase not in self._hooks:
            raise ValueError(f"Unknown lifecycle phase: {phase}")
        self._hooks[phase].append(hook)
    
    def unregister(self, phase: str, hook: LifecycleHook) -> None:
        """注销生命周期钩子"""
        if phase in self._hooks and hook in self._hooks[phase]:
            self._hooks[phase].remove(hook)
    
    def clear(self) -> None:
        """清空所有钩子"""
        for phase in self._hooks:
            self._hooks[phase].clear()
    
    def get_hooks(self, phase: str) -> list[LifecycleHook]:
        """获取指定阶段的所有钩子"""
        return self._hooks.get(phase, [])
    
    async def invoke(self, phase: str, context: Any) -> None:
        """调用指定阶段的所有钩子"""
        import asyncio
        hooks = self.get_hooks(phase)
        for hook in hooks:
            try:
                result = hook(context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # 钩子异常不应该阻止执行
                import logging
                logging.getLogger(__name__).warning(
                    f"Error in {phase} hook: {e}"
                )


def get_lifecycle_hooks() -> LifecycleHooks:
    """获取全局生命周期钩子管理器"""
    return LifecycleHooks()


def register_hook(phase: str, hook: LifecycleHook) -> None:
    """注册生命周期钩子的便捷函数"""
    get_lifecycle_hooks().register(phase, hook)


def unregister_hook(phase: str, hook: LifecycleHook) -> None:
    """注销生命周期钩子的便捷函数"""
    get_lifecycle_hooks().unregister(phase, hook)


async def invoke_hooks(phase: str, context: Any) -> None:
    """调用生命周期钩子的便捷函数"""
    await get_lifecycle_hooks().invoke(phase, context)
