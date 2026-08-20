"""Drift 空闲任务 —— 没内容可推时，Agent 自主选择并执行后台 skill。

流程：
  1. Scan     —— 扫描 workspace/skills/ 目录，收集 SKILL.md 列表
  2. Select   —— 让 LLM 从可用 skill 中选一个执行（或判断无事可做）
  3. Execute  —— 把 SKILL.md 内容作为指导，让 LLM 自主完成步骤
  4. Finish   —— 记录执行结果
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from agent.provider import LLMProvider

log = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"


@dataclass
class SkillEntry:
    """一个可执行的后台 skill。"""

    name: str
    path: Path
    description: str = ""


@dataclass
class DriftResult:
    """一轮 Drift 的执行结果。"""

    action: str  # "executed" | "idle" | "no_skills"
    skill_name: str = ""
    summary: str = ""
    timestamp: str = ""


class DriftLoop:
    """空闲时自主执行后台任务的循环。"""

    def __init__(
        self,
        provider: LLMProvider,
        skills_dir: Path,
    ) -> None:
        self._provider = provider
        self._skills_dir = skills_dir

    def scan_skills(self) -> list[SkillEntry]:
        """扫描 skills 目录，返回所有包含 SKILL.md 的子目录。"""
        if not self._skills_dir.exists():
            return []
        skills: list[SkillEntry] = []
        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / SKILL_FILENAME
            if not skill_file.exists():
                continue
            description = _extract_first_heading(skill_file.read_text(encoding="utf-8"))
            skills.append(
                SkillEntry(name=child.name, path=skill_file, description=description)
            )
        return skills

    async def run(self) -> DriftResult:
        """执行一轮 Drift：扫描 skill → LLM 选择 → 执行。"""
        skills = self.scan_skills()
        if not skills:
            return DriftResult(action="no_skills", timestamp=_now_iso())

        skill_list_text = "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )

        select_prompt = (
            "你是一个 AI Agent，现在处于空闲时间，可以自主选择一个后台任务执行。\n"
            "以下是可用的 skill 列表。请选择一个最适合当前执行的任务，"
            "或者判断没有需要执行的任务。\n\n"
            f"可用 skill：\n{skill_list_text}\n\n"
            "只输出 skill 名称，或输出 NONE 表示无事可做。"
        )

        response = await self._provider.chat(
            [
                {"role": "system", "content": "你是 ProactiveMind 的 Drift 子系统。"},
                {"role": "user", "content": select_prompt},
            ],
            max_tokens=100,
        )

        choice = response.content.strip()
        if choice.upper() == "NONE" or not choice:
            return DriftResult(
                action="idle", timestamp=_now_iso(), summary="LLM 判断无事可做"
            )

        selected = next((s for s in skills if s.name == choice), None)
        if selected is None:
            return DriftResult(
                action="idle",
                timestamp=_now_iso(),
                summary=f"LLM 选择了不存在的 skill: {choice}",
            )

        return await self._execute_skill(selected)

    async def _execute_skill(self, skill: SkillEntry) -> DriftResult:
        """读取 SKILL.md 内容并让 LLM 自主执行。"""
        skill_content = skill.path.read_text(encoding="utf-8")

        execute_prompt = (
            "请按照以下 skill 指南执行任务。使用可用工具完成步骤，"
            "完成后给出简要总结。\n\n"
            f"--- {skill.name} SKILL.md ---\n{skill_content}"
        )

        response = await self._provider.chat(
            [
                {
                    "role": "system",
                    "content": "你是 ProactiveMind 的 Drift 子系统，正在执行后台任务。",
                },
                {"role": "user", "content": execute_prompt},
            ],
        )

        log.info("Drift 执行 skill=%s", skill.name)
        return DriftResult(
            action="executed",
            skill_name=skill.name,
            summary=response.content[:500],
            timestamp=_now_iso(),
        )


def _extract_first_heading(markdown: str) -> str:
    """从 Markdown 中提取第一个标题作为描述。"""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
