"""配置加载。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096


@dataclass
class Config:
    llm: LLMConfig
    workspace: Path


def load_config(path: str = "config.toml") -> Config:
    """从 TOML 文件加载配置，支持 ${ENV_VAR} 插值。"""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件 {path}，请参考 config.example.toml 创建"
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

    return Config(
        llm=LLMConfig(
            api_key=api_key,
            base_url=llm_section.get("base_url", "https://api.openai.com/v1"),
            model=llm_section.get("model", "gpt-4o"),
            max_tokens=llm_section.get("max_tokens", 4096),
        ),
        workspace=Path(workspace_str).expanduser().resolve(),
    )
