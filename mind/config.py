"""配置加载。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
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
    llm: LLMConfig
    workspace: Path
    max_history_tokens: int = 6000
    prompt: PromptConfig = field(default_factory=PromptConfig)
    consolidation: ConsolidationConfig = field(
        default_factory=ConsolidationConfig
    )
    extensions_dir: Path | None = None


def load_config(path: str = "config.toml") -> Config:
    """从 TOML 文件加载配置，支持 ${ENV_VAR} 插值。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {path}，请参考 config.sample.toml 创建"
        )

    raw = config_path.read_text(encoding="utf-8")
    # 简单的环境变量插值：${VAR} → os.environ["VAR"]
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

    llm_section = data.get("llm", {})
    api_key = llm_section.get("api_key", "")
    if not api_key:
        raise ValueError("[llm].api_key 不能为空")

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

    extensions_dir_str = extensions_section.get("dir", "")
    extensions_dir = (
        Path(extensions_dir_str).expanduser().resolve()
        if extensions_dir_str
        else None
    )

    return Config(
        llm=LLMConfig(
            api_key=api_key,
            base_url=llm_section.get("base_url", "https://api.openai.com/v1"),
            model=llm_section.get("model", "gpt-4o"),
            max_tokens=llm_section.get("max_tokens", 4096),
        ),
        workspace=Path(workspace_str).expanduser().resolve(),
        max_history_tokens=context_section.get("max_history_tokens", 6000),
        prompt=PromptConfig(
            persona=prompt_section.get("persona", PromptConfig().persona),
            rules=rules,
        ),
        consolidation=ConsolidationConfig(
            enabled=consolidation_enabled
        ),
        extensions_dir=extensions_dir,
    )


def validate_config(config: Config) -> list[str]:
    """校验配置，返回问题列表（空列表表示无问题）。"""
    problems: list[str] = []

    if config.max_history_tokens < 100:
        problems.append("max_history_tokens 过小（建议至少 100）")

    if config.max_history_tokens > 100000:
        problems.append("max_history_tokens 过大（建议不超过 100000）")

    if config.llm.max_tokens < 100:
        problems.append("llm.max_tokens 过小（建议至少 100）")

    if not config.llm.api_key:
        problems.append("llm.api_key 为空")

    if not config.llm.base_url:
        problems.append("llm.base_url 为空")

    if not config.llm.model:
        problems.append("llm.model 为空")

    if config.extensions_dir is not None and not config.extensions_dir.exists():
        problems.append(f"extensions_dir 不存在: {config.extensions_dir}")

    return problems
