"""自动记忆归档测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mind.consolidation import MemoryConsolidator
from mind.memory import MemoryStore
from mind.provider import LLMResponse


class FakeProvider:
    """返回预设候选记忆的 Provider。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict] = []
        self.max_tokens: int | None = None

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.messages = messages
        self.max_tokens = max_tokens
        return LLMResponse(content=self.content)


class MemoryConsolidatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_stages_extracted_facts_without_changing_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryStore(Path(temp_dir))
            provider = FakeProvider(
                '{"facts":["用户长期维护 ProactiveMind 项目",'
                '"偏好使用 Python 3.12"]}'
            )
            consolidator = MemoryConsolidator(provider, memory)

            facts = await consolidator.consolidate("我会长期维护项目", "好的")

            self.assertEqual(
                facts,
                ["用户长期维护 ProactiveMind 项目", "偏好使用 Python 3.12"],
            )
            self.assertEqual(provider.max_tokens, 400)
            self.assertIn("用户：我会长期维护项目", provider.messages[1]["content"])
            self.assertNotIn("ProactiveMind 项目", memory.read_all())
            pending = (Path(temp_dir) / "PENDING.md").read_text(encoding="utf-8")
            self.assertIn("用户长期维护 ProactiveMind 项目", pending)

    async def test_ignores_invalid_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = MemoryStore(Path(temp_dir))
            consolidator = MemoryConsolidator(FakeProvider("不是 JSON"), memory)

            facts = await consolidator.consolidate("今天天气如何", "晴朗")

            self.assertEqual(facts, [])
            pending = (Path(temp_dir) / "PENDING.md").read_text(encoding="utf-8")
            self.assertNotIn("今天天气如何", pending)

    def test_limits_and_deduplicates_facts(self) -> None:
        content = (
            '{"facts":["  长期偏好  ", "长期偏好", 1, "", '
            '"a", "b", "c", "d", "e"]}'
        )

        facts = MemoryConsolidator._parse_facts(content)

        self.assertEqual(facts, ["长期偏好", "a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
