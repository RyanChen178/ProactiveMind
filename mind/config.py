"""配置加载。"""

from __future__ import annotations

import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path

from mind.runtime import RuntimeConfig, RuntimeRegistry, load_runtime_registry


@dataclass(frozen=True)
class ContextCompactionConfig:
    """上下文压缩配置。"""

    keep_recent_tokens: int = 20000

    def __post_init__(self) -> None:
        if self.keep_recent_tokens <= 0:
            raise ValueError("context.compaction.keep_recent_tokens 必须是正整数")


@dataclass
class LLMConfig:
    """向后兼容的旧版 LLM 配置（仅用于迁移期）。"""

    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096


@dataclass
class PromptConfig:
    persona: str = "你是 ProactiveMind，一个有持久记忆的 AI 助手。"
    rules: list[str] = field(
        default_factory=lambda: [
            "准确、诚实地回答；不确定时说明不确定性。",
            "使用工具前先判断是否确有必要。",
            "重要且长期有效的用户事实可使用 memorize 保存。",
        ]
    )


@dataclass
class ConsolidationConfig:
    enabled: bool = True


@dataclass
class Config:
    runtimes: RuntimeRegistry
    workspace: Path
    max_history_tokens: int = 6000
    context_compaction: ContextCompactionConfig = field(
        default_factory=ContextCompactionConfig
    )
    prompt: PromptConfig = field(default_factory=PromptConfig)
    consolidation: ConsolidationConfig = field(
        default_factory=ConsolidationConfig
    )
    extensions_dir: Path | None = None

    @property
    def llm(self) -> LLMConfig:
        """向后兼容属性——返回主运行时的 LLM 配置。"""
        main_runtime = self.runtimes.get_main()
        return LLMConfig(
            api_key=main_runtime.api_key,
            base_url=main_runtime.base_url,
            model=main_runtime.model,
            max_tokens=main_runtime.max_output_tokens or 4096,
        )


def load_config(path: str = "config.toml") -> Config:
    """从 TOML 文件加载配置，支持 ${ENV_VAR} 插值。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {path}，请参考 config.sample.toml 创建"
        )

    raw = config_path.read_text(encoding="utf-8")
    for _ in range(8):
        start = raw.find("${")
        if start == -1:
            break
        end = raw.find("}", start)
        if end == -1:
            break
        var_name = raw[start + 2 : end]
        env_value = os.environ.get(var_name, "")
        raw = raw[:start] + env_value + raw[end + 1 :]

    data = tomllib.loads(raw)

    # 加载运行时注册表
    runtimes = load_runtime_registry(data)

    # 向后兼容：如果没有配置运行时，尝试旧格式
    if not runtimes.runtimes:
        llm_section = data.get("llm", {})
        api_key = llm_section.get("api_key", "")
        if not api_key:
            raise ValueError("必须配置 [llm.runtimes.xxx] 或旧版 [llm]")

        base_url = llm_section.get("base_url", "https://api.openai.com/v1")
        model = llm_section.get("model", "gpt-4o")
        max_tokens = llm_section.get("max_tokens", 4096)

        default_runtime = RuntimeConfig(
            runtime_id="default",
            provider=_infer_provider(base_url),
            model=model,
            api_key=api_key,
            base_url=base_url,
            context_window=128000,
            max_output_tokens=max_tokens,
        )
        runtimes = RuntimeRegistry(
            runtimes={"default": default_runtime},
            main="default",
        )

    workspace_str = data.get("workspace", {}).get("path", "~/.proactivemind/workspace")
    context_section = data.get("context", {})
    prompt_section = data.get("prompt", {})
    consolidation_section = data.get("consolidation", {})
    extensions_section = data.get("extensions", {})
    rules = prompt_section.get("rules", PromptConfig().rules)
    consolidation_enabled = consolidation_section.get("enabled", True)

    if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
        raise ValueError("[prompt].rules 必须是字符串列表")
    if not isinstance(consolidation_enabled, bool):
        raise ValueError("[consolidation].enabled 必须是布尔值")

    compaction_section = context_section.get("compaction", {})
    keep_recent_tokens = compaction_section.get("keep_recent_tokens", 20000)

    extensions_dir_str = extensions_section.get("dir", "")
    extensions_dir = (
        Path(extensions_dir_str).expanduser().resolve()
        if extensions_dir_str
        else None
    )

    return Config(
        runtimes=runtimes,
        workspace=Path(workspace_str).expanduser().resolve(),
        max_history_tokens=context_section.get("max_history_tokens", 6000),
        context_compaction=ContextCompactionConfig(
            keep_recent_tokens=keep_recent_tokens
        ),
        prompt=PromptConfig(
            persona=prompt_section.get("persona", PromptConfig().persona),
            rules=rules,
        ),
        consolidation=ConsolidationConfig(
            enabled=consolidation_enabled
        ),
        extensions_dir=extensions_dir,
    )


def _infer_provider(base_url: str) -> str:
    """根据 base_url 推断 provider 名称。"""
    if "deepseek" in base_url:
        return "deepseek"
    if "dashscope" in base_url or "qwen" in base_url:
        return "qwen"
    if "openai" in base_url:
        return "openai"
    return "openai"


def validate_config(config: Config) -> list[str]:
    """校验配置，返回问题列表（空列表表示无问题）。"""
    problems: list[str] = []

    if config.max_history_tokens < 100:
        problems.append("max_history_tokens 过小（建议至少 100）")

    if config.max_history_tokens > 100000:
        problems.append("max_history_tokens 过大（建议不超过 100000）")

    if not config.runtimes.runtimes:
        problems.append("未配置任何运行时")
    elif not config.runtimes.main:
        problems.append("未指定 main 运行时")
    else:
        try:
            main_runtime = config.runtimes.get_main()
            if not main_runtime.api_key:
                problems.append("main 运行时的 api_key 为空")
            if not main_runtime.base_url:
                problems.append("main 运行时的 base_url 为空")
            if not main_runtime.model:
                problems.append("main 运行时的 model 为空")
        except ValueError as e:
            problems.append(str(e))

    if config.extensions_dir is not None and not config.extensions_dir.exists():
        problems.append(f"extensions_dir 不存在: {config.extensions_dir}")

    return problems
