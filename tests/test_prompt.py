"""系统提示词组装测试。"""

from __future__ import annotations

import unittest

from agent.config import PromptConfig
from agent.prompt import PromptBuilder


class PromptBuilderTest(unittest.TestCase):
    def test_builds_layers_in_fixed_order(self) -> None:
        prompt = PromptBuilder(
            PromptConfig(persona="耐心的助手", rules=["先验证，再回答"])
        ).build("用户偏好中文")

        self.assertLess(prompt.index("## 人格"), prompt.index("## 行为规则"))
        self.assertLess(prompt.index("## 行为规则"), prompt.index("## 工具说明"))
        self.assertLess(prompt.index("## 工具说明"), prompt.index("## 已有记忆"))
        self.assertIn("耐心的助手", prompt)
        self.assertIn("- 先验证，再回答", prompt)
        self.assertIn("用户偏好中文", prompt)

    def test_omits_empty_memory_layer(self) -> None:
        prompt = PromptBuilder(PromptConfig()).build()

        self.assertNotIn("## 已有记忆", prompt)


if __name__ == "__main__":
    unittest.main()
