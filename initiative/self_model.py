"""自我模型 —— Agent 维护自我认知文件 (Self.md)。

Self.md 记录 Agent 的：
- 偏好（preferences）：回复风格、工具选择倾向等
- 能力边界（capabilities）：已知限制、擅长领域
- 当前目标（goals）：正在进行的任务、关注点

Agent 可以在 Wander 空闲任务中更新 Self.md，逐步完善自我认知。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SelfModel:
    """自我模型，记录 Agent 的自我认知。"""
    
    preferences: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    updated_at: datetime | None = None
    
    def add_preference(self, preference: str) -> None:
        """添加偏好。"""
        if preference and preference not in self.preferences:
            self.preferences.append(preference)
            self.updated_at = datetime.now(timezone.utc)
    
    def add_capability(self, capability: str) -> None:
        """添加能力描述。"""
        if capability and capability not in self.capabilities:
            self.capabilities.append(capability)
            self.updated_at = datetime.now(timezone.utc)
    
    def add_goal(self, goal: str) -> None:
        """添加当前目标。"""
        if goal and goal not in self.goals:
            self.goals.append(goal)
            self.updated_at = datetime.now(timezone.utc)
    
    def remove_goal(self, goal: str) -> None:
        """移除目标（完成后移除）。"""
        if goal in self.goals:
            self.goals.remove(goal)
            self.updated_at = datetime.now(timezone.utc)
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式。"""
        lines = ["# Self Model\n"]
        
        if self.preferences:
            lines.append("## Preferences\n")
            for pref in self.preferences:
                lines.append(f"- {pref}")
            lines.append("")
        
        if self.capabilities:
            lines.append("## Capabilities\n")
            for cap in self.capabilities:
                lines.append(f"- {cap}")
            lines.append("")
        
        if self.goals:
            lines.append("## Goals\n")
            for goal in self.goals:
                lines.append(f"- {goal}")
            lines.append("")
        
        if self.updated_at:
            lines.append(f"*Last updated: {self.updated_at.isoformat()}*")
        
        return "\n".join(lines)
    
    @classmethod
    def from_markdown(cls, content: str) -> SelfModel:
        """从 Markdown 解析自我模型。"""
        model = cls()
        current_section = None
        
        for line in content.split("\n"):
            line = line.strip()
            
            if line.startswith("## "):
                section = line[3:].lower()
                if section == "preferences":
                    current_section = "preferences"
                elif section == "capabilities":
                    current_section = "capabilities"
                elif section == "goals":
                    current_section = "goals"
                else:
                    current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:]
                if current_section == "preferences":
                    model.preferences.append(item)
                elif current_section == "capabilities":
                    model.capabilities.append(item)
                elif current_section == "goals":
                    model.goals.append(item)
            elif line.startswith("*Last updated:"):
                # 解析时间戳
                import re
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', line)
                if match:
                    try:
                        model.updated_at = datetime.fromisoformat(match.group(0))
                    except ValueError:
                        pass
        
        return model


class SelfModelManager:
    """自我模型管理器，负责读写 Self.md。"""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.self_file = workspace / "Self.md"
        self._model: SelfModel | None = None
    
    def load(self) -> SelfModel:
        """加载自我模型。"""
        if self.self_file.exists():
            content = self.self_file.read_text(encoding="utf-8")
            self._model = SelfModel.from_markdown(content)
            log.info("Loaded Self.md: %d preferences, %d capabilities, %d goals",
                    len(self._model.preferences),
                    len(self._model.capabilities),
                    len(self._model.goals))
        else:
            self._model = SelfModel()
            log.info("Created new Self.md")
        
        return self._model
    
    def save(self, model: SelfModel | None = None) -> None:
        """保存自我模型到 Self.md。"""
        if model is not None:
            self._model = model
        
        if self._model is None:
            self._model = SelfModel()
        
        content = self._model.to_markdown()
        self.self_file.write_text(content, encoding="utf-8")
        log.info("Saved Self.md: %d preferences, %d capabilities, %d goals",
                len(self._model.preferences),
                len(self._model.capabilities),
                len(self._model.goals))
    
    def get_model(self) -> SelfModel:
        """获取当前自我模型。"""
        if self._model is None:
            return self.load()
        return self._model
    
    def update_from_drift(self, drift_summary: str) -> None:
        """根据 Drift 执行结果更新自我模型。
        
        从 Drift 的 summary 中提取可能的自我认知更新：
        - 如果 summary 包含"学到了"、"发现"等关键词，可能是能力更新
        - 如果 summary 包含"应该"、"需要"等关键词，可能是目标更新
        """
        model = self.get_model()
        
        # 简单的关键词匹配（后续可以用 LLM 提取）
        keywords_capability = ["学到了", "学会了", "掌握了", "能够", "可以"]
        keywords_goal = ["应该", "需要", "必须", "计划", "目标"]
        
        summary_lower = drift_summary.lower()
        
        # 检查是否有能力更新
        if any(kw in drift_summary for kw in keywords_capability):
            model.add_capability(drift_summary[:100])  # 限制长度
            log.info("Added capability from Drift: %s", drift_summary[:50])
        
        # 检查是否有目标更新
        if any(kw in drift_summary for kw in keywords_goal):
            model.add_goal(drift_summary[:100])  # 限制长度
            log.info("Added goal from Drift: %s", drift_summary[:50])
        
        self.save()
