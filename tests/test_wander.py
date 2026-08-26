"""Drift 空闲任务测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from initiative.drift import WanderLoop
from mind.provider import LLMResponse


class FakeProvider:
    """按预设返回模型响应。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, max_tokens=None) -> LLMResponse:
        self.calls.append(messages)
        return next(self._responses)


class WanderLoopTest(unittest.IsolatedAsyncioTestCase):
    def _make_skills_dir(self, temp_dir: str) -> Path:
        skills_dir = Path(temp_dir) / "skills"
        (skills_dir / "audit-memory").mkdir(parents=True)
        (skills_dir / "audit-memory" / "PLAYBOOK.md").write_text(
            "# 审计长期记忆\n\n检查 MEMORY.md", encoding="utf-8"
        )
        (skills_dir / "self-check").mkdir(parents=True)
        (skills_dir / "self-check" / "PLAYBOOK.md").write_text(
            "# 自我诊断\n\n检查系统状态", encoding="utf-8"
        )
        return skills_dir

    def test_scan_finds_all_skills_with_md(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = self._make_skills_dir(temp_dir)
            drift = WanderLoop(FakeProvider([]), skills_dir)

            skills = drift.scan_playbooks()

            names = [s.name for s in skills]
            self.assertEqual(names, ["audit-memory", "self-check"])
            self.assertEqual(skills[0].description, "审计长期记忆")
            self.assertEqual(skills[1].description, "自我诊断")

    def test_scan_returns_empty_when_no_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            drift = WanderLoop(FakeProvider([]), Path(temp_dir) / "skills")

            self.assertEqual(drift.scan_playbooks(), [])

    async def test_returns_no_skills_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            drift = WanderLoop(FakeProvider([]), Path(temp_dir) / "skills")

            result = await drift.run()

            self.assertEqual(result.action, "no_skills")

    async def test_executes_selected_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = self._make_skills_dir(temp_dir)
            provider = FakeProvider([
                LLMResponse(content="audit-memory"),
                LLMResponse(content="审计完成，发现 2 条过时记忆"),
            ])
            drift = WanderLoop(provider, skills_dir)

            result = await drift.run()

            self.assertEqual(result.action, "executed")
            self.assertEqual(result.skill_name, "audit-memory")
            self.assertIn("审计完成", result.summary)
            self.assertEqual(len(provider.calls), 2)

    async def test_returns_idle_when_llm_says_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = self._make_skills_dir(temp_dir)
            provider = FakeProvider([LLMResponse(content="NONE")])
            drift = WanderLoop(provider, skills_dir)

            result = await drift.run()

            self.assertEqual(result.action, "idle")
            self.assertEqual(len(provider.calls), 1)

    async def test_returns_idle_when_skill_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = self._make_skills_dir(temp_dir)
            provider = FakeProvider([LLMResponse(content="nonexistent-skill")])
            drift = WanderLoop(provider, skills_dir)

            result = await drift.run()

            self.assertEqual(result.action, "idle")
            self.assertIn("nonexistent-skill", result.summary)


if __name__ == "__main__":
    unittest.main()
