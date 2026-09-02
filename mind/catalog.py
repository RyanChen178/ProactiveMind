"""模型能力目录 - 提供常见模型的能力信息（上下文窗口、视觉支持等）。

当配置中未指定 context_window 时，从目录中查找默认值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapability:
    """模型能力信息。"""
    
    context_window: int
    supports_vision: bool = False
    max_output_tokens: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "context_window": self.context_window,
            "supports_vision": self.supports_vision,
            "max_output_tokens": self.max_output_tokens,
        }


# 常见模型的能力目录
_MODEL_CATALOG = {
    # DeepSeek 系列
    "deepseek/deepseek-chat": ModelCapability(context_window=64000),
    "deepseek/deepseek-coder": ModelCapability(context_window=64000),
    "deepseek/deepseek-v2": ModelCapability(context_window=128000),
    "deepseek/deepseek-v2.5": ModelCapability(context_window=128000),
    "deepseek/deepseek-v3": ModelCapability(context_window=128000),
    
    # Qwen 系列
    "qwen/qwen-turbo": ModelCapability(context_window=128000),
    "qwen/qwen-plus": ModelCapability(context_window=128000),
    "qwen/qwen-max": ModelCapability(context_window=32000),
    "qwen/qwen-vl-plus": ModelCapability(context_window=32000, supports_vision=True),
    "qwen/qwen-vl-max": ModelCapability(context_window=32000, supports_vision=True),
    
    # OpenAI 系列
    "openai/gpt-4": ModelCapability(context_window=8192),
    "openai/gpt-4-turbo": ModelCapability(context_window=128000),
    "openai/gpt-4o": ModelCapability(context_window=128000, supports_vision=True),
    "openai/gpt-4o-mini": ModelCapability(context_window=128000, supports_vision=True),
    "openai/gpt-3.5-turbo": ModelCapability(context_window=16385),
    
    # Anthropic 系列
    "anthropic/claude-3-opus": ModelCapability(context_window=200000, supports_vision=True),
    "anthropic/claude-3-sonnet": ModelCapability(context_window=200000, supports_vision=True),
    "anthropic/claude-3-haiku": ModelCapability(context_window=200000, supports_vision=True),
    "anthropic/claude-3.5-sonnet": ModelCapability(context_window=200000, supports_vision=True),
    
    # Google 系列
    "google/gemini-pro": ModelCapability(context_window=32760),
    "google/gemini-1.5-pro": ModelCapability(context_window=1000000, supports_vision=True),
    "google/gemini-1.5-flash": ModelCapability(context_window=1000000, supports_vision=True),
}


def lookup_model_capability(
    provider: str,
    model: str,
) -> Optional[ModelCapability]:
    """查找模型能力信息。
    
    Args:
        provider: 提供商名称（如 "deepseek", "qwen", "openai"）
        model: 模型名称（如 "deepseek-chat", "gpt-4"）
    
    Returns:
        ModelCapability 如果找到，否则 None
    """
    # 尝试精确匹配
    key = f"{provider}/{model}"
    if key in _MODEL_CATALOG:
        return _MODEL_CATALOG[key]
    
    # 尝试只匹配模型名
    for catalog_key, capability in _MODEL_CATALOG.items():
        if catalog_key.endswith(f"/{model}"):
            return capability
    
    # 未找到
    logger.debug(f"模型 {key} 未在能力目录中找到")
    return None


def get_context_window(
    provider: str,
    model: str,
    default: int = 128000,
) -> int:
    """获取模型的上下文窗口大小。
    
    Args:
        provider: 提供商名称
        model: 模型名称
        default: 如果未找到，返回的默认值
    
    Returns:
        上下文窗口大小（token 数）
    """
    capability = lookup_model_capability(provider, model)
    if capability:
        return capability.context_window
    return default


def get_max_output_tokens(
    provider: str,
    model: str,
    default: int = 0,
) -> int:
    """获取模型的最大输出 token 数。
    
    Args:
        provider: 提供商名称
        model: 模型名称
        default: 如果未找到，返回的默认值（0 表示不限制）
    
    Returns:
        最大输出 token 数
    """
    capability = lookup_model_capability(provider, model)
    if capability and capability.max_output_tokens is not None:
        return capability.max_output_tokens
    return default


def supports_vision(
    provider: str,
    model: str,
) -> bool:
    """检查模型是否支持视觉输入。
    
    Args:
        provider: 提供商名称
        model: 模型名称
    
    Returns:
        是否支持视觉输入
    """
    capability = lookup_model_capability(provider, model)
    if capability:
        return capability.supports_vision
    return False
