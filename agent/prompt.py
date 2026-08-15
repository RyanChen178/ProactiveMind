"""系统提示词分层组装。"""

from __future__ import annotations

from agent.config import PromptConfig

TOOL_GUIDE = """你可以使用工具帮助用户：
- get_time：获取当前时间
- shell：执行 shell 命令
- memorize：保存重要事实到长期记忆
- recall：检索长期记忆"""


class PromptBuilder:
    """将稳定规则与动态记忆组装为系统提示词。"""

    def __init__(self, config: PromptConfig) -> None:
        self._config = config

    def build(self, memory_text: str = "") -> str:
        """按固定顺序输出各个提示词区块。"""

        blocks = [
            f"## 人格\n{self._config.persona}",
            "## 行为规则\n" + "\n".join(
                f"- {rule}" for rule in self._config.rules
            ),
            f"## 工具说明\n{TOOL_GUIDE}",
        ]
        if memory_text.strip():
            blocks.append(f"## 已有记忆\n{memory_text.strip()}")
        return "\n\n".join(blocks)
