"""多模型运行时系统——支持命名运行时与多模型切换。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeConfig:
    """单个 LLM 运行时配置。"""

    runtime_id: str
    provider: str
    model: str
    api_key: str
    base_url: str
    context_window: int = 128000
    max_output_tokens: int = 0
    input_modalities: tuple[str, ...] = ('text',)
    reasoning_effort: str = ''
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise ValueError(f'运行时 {self.runtime_id} 必须配置 provider 和 model')
        if not self.base_url:
            raise ValueError(f'运行时 {self.runtime_id} 的 base_url 不能为空')
        if self.context_window <= 0:
            raise ValueError(f'运行时 {self.runtime_id} 的 context_window 必须为正整数')
        if self.max_output_tokens < 0:
            raise ValueError(f'运行时 {self.runtime_id} 的 max_output_tokens 不能为负')
        if self.max_output_tokens >= self.context_window:
            raise ValueError(
                f'运行时 {self.runtime_id} 的 max_output_tokens 必须小于 context_window'
            )
        if 'text' not in self.input_modalities:
            raise ValueError(f'运行时 {self.runtime_id} 的 input_modalities 必须包含 text')


@dataclass(frozen=True)
class RuntimeRegistry:
    """运行时注册表——管理多个命名运行时。"""

    runtimes: dict[str, RuntimeConfig] = field(default_factory=dict)
    main: str = ''
    fast: str = ''
    vl: str = ''

    def __post_init__(self) -> None:
        if self.main and self.main not in self.runtimes:
            raise ValueError(f'main 运行时 {self.main} 未定义')
        if self.fast and self.fast not in self.runtimes:
            raise ValueError(f'fast 运行时 {self.fast} 未定义')
        if self.vl and self.vl not in self.runtimes:
            raise ValueError(f'vl 运行时 {self.vl} 未定义')

    def get(self, name: str) -> RuntimeConfig | None:
        return self.runtimes.get(name)

    def get_main(self) -> RuntimeConfig:
        if not self.main:
            raise ValueError('未配置 main 运行时')
        runtime = self.runtimes.get(self.main)
        if not runtime:
            raise ValueError(f'main 运行时 {self.main} 未定义')
        return runtime

    def get_fast(self) -> RuntimeConfig | None:
        if not self.fast:
            return None
        return self.runtimes.get(self.fast)

    def get_vl(self) -> RuntimeConfig | None:
        if not self.vl:
            return None
        return self.runtimes.get(self.vl)

    def list_names(self) -> list[str]:
        return sorted(self.runtimes.keys())


def load_runtime_registry(data: dict) -> RuntimeRegistry:
    """从 TOML 数据加载运行时注册表。"""
    llm_section = data.get('llm', {})
    main_name = llm_section.get('main', '')
    fast_name = llm_section.get('fast', '')
    vl_name = llm_section.get('vl', '')
    runtimes_section = llm_section.get('runtimes', {})
    runtimes: dict[str, RuntimeConfig] = {}

    for runtime_id, runtime_data in runtimes_section.items():
        if not isinstance(runtime_data, dict):
            continue
        provider = runtime_data.get('provider', '')
        model = runtime_data.get('model', '')
        api_key = _expand_env_var(runtime_data.get('api_key', ''))
        base_url = _expand_env_var(runtime_data.get('base_url', ''))
        context_window = runtime_data.get('context_window', 128000)
        max_output_tokens = runtime_data.get('max_output_tokens', 0)
        input_modalities_raw = runtime_data.get('input_modalities', ['text'])
        if isinstance(input_modalities_raw, list):
            input_modalities = tuple(str(m) for m in input_modalities_raw)
        else:
            input_modalities = ('text',)
        reasoning_effort = runtime_data.get('reasoning_effort', '')
        enable_thinking = runtime_data.get('enable_thinking', False)
        runtimes[runtime_id] = RuntimeConfig(
            runtime_id=runtime_id,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            input_modalities=input_modalities,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )

    return RuntimeRegistry(
        runtimes=runtimes, main=main_name, fast=fast_name, vl=vl_name,
    )


def _expand_env_var(value: str) -> str:
    if not isinstance(value, str):
        return value
    result = value
    for _ in range(8):
        start = result.find('${')
        if start == -1:
            break
        end = result.find('}', start)
        if end == -1:
            break
        var_name = result[start + 2: end]
        env_value = os.environ.get(var_name, '')
        result = result[:start] + env_value + result[end + 1:]
    return result
