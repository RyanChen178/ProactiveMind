"""主动链路漂移状态 —— skill 扫描、运行记录与全局漂移历史。

设计思想（与 a20b45af proactive_v2/drift_state.py 对齐）：

  - Skill 扫描：遍历 skills 目录（可选 builtin 目录），读每个
    skill 的 SKILL.md frontmatter（name / description / requires_mcp）
    + state.json（last_run_at / run_count / status / next）
  - 去重与排序：同名 skill 只保留第一个来源；按最近运行时间倒序
  - save_finish：写 skill 级 state.json（run_count+1、status、next）
    + 全局 drift.json 的 recent_runs（滚动保留最近 10 条）
  - load_drift 清洗：字段截断、message_result 只允许 sent/silent、
    缺关键字段的行丢弃
  - skill_dir_for：先查工作区 skills 目录，再查 builtin 目录

frontmatter 解析用纯正则（key: value + YAML 列表项），不引入
YAML 依赖；持久化用 json + 临时文件原子替换。

ProactiveMind 命名（initiative/proactive_drift_state.py）独立于原仓库。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECENT_RUNS_LIMIT = 10
SKILL_FILENAME = "SKILL.md"
STATE_FILENAME = "state.json"
DRIFT_FILENAME = "drift.json"
_VALID_STATUS = frozenset({"idle", "in_progress"})
_VALID_MESSAGE_RESULTS = frozenset({"sent", "silent"})


def _clip(text: Any, limit: int) -> str:
    return str(text or "").strip()[:limit]


def _parse_iso(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


def parse_frontmatter(content: str) -> dict[str, str | list[str]]:
    """解析 Markdown frontmatter（--- 包围的 YAML 子集）。

    支持两种值形态：
      key: value           —— 标量
      key:                 —— 空值后跟 "- item" 列表项
    """
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match is None:
        return {}

    metadata: dict[str, str | list[str]] = {}
    lines = match.group(1).split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if value:
            metadata[key] = value
            i += 1
            continue
        # value 为空 → 后续行是否为列表项（- item）
        items: list[str] = []
        j = i + 1
        while j < len(lines):
            item_match = re.match(r"^\s+-\s+(.+)", lines[j])
            if item_match is None:
                break
            items.append(item_match.group(1).strip().strip("\"'"))
            j += 1
        if items:
            metadata[key] = items
            i = j
        else:
            metadata[key] = value
            i += 1
    return metadata


@dataclass
class SkillMeta:
    """一个漂移 skill 的元信息（SKILL.md + state.json 合成）。"""

    name: str
    description: str
    last_run_at: datetime | None = None
    run_count: int = 0
    status: str = "idle"
    next: str = ""
    requires_mcp: list[str] = field(default_factory=list)
    builtin: bool = False


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再原子替换，避免半写状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[drift_state] 读取失败 %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


class DriftStateStore:
    """漂移 skill 状态仓库。"""

    def __init__(
        self,
        drift_dir: Path,
        *,
        builtin_skills_dir: Path | None = None,
        include_builtin_skills: bool = False,
        builtin_skill_names: set[str] | None = None,
    ) -> None:
        self.drift_dir = Path(drift_dir).expanduser()
        self.skills_dir = self.drift_dir / "skills"
        self.drift_file = self.drift_dir / DRIFT_FILENAME
        self.builtin_skills_dir = (
            Path(builtin_skills_dir).expanduser()
            if builtin_skills_dir is not None
            else None
        )
        self.include_builtin_skills = include_builtin_skills
        self.builtin_skill_names = set(builtin_skill_names or set())
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------

    def _skill_roots(self) -> list[tuple[Path, bool]]:
        roots: list[tuple[Path, bool]] = [(self.skills_dir, False)]
        if self.include_builtin_skills and self.builtin_skills_dir is not None:
            roots.append((self.builtin_skills_dir, True))
        return roots

    @staticmethod
    def _normalize_status(raw: Any) -> str:
        status = str(raw or "").strip()
        return status if status in _VALID_STATUS else "idle"

    def _load_skill_meta(self, skill_dir: Path, *, builtin: bool) -> SkillMeta | None:
        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.exists():
            return None
        metadata = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        name = str(metadata.get("name") or "").strip()
        description = str(metadata.get("description") or "").strip()
        # name 必须非空、描述非空、且与目录名一致（防止目录与内容错位）
        if not name or not description or name != skill_dir.name:
            logger.info(
                "[drift_state] 跳过无效 skill dir=%s name=%r", skill_dir, name
            )
            return None

        requires_mcp_val = metadata.get("requires_mcp")
        if isinstance(requires_mcp_val, list):
            requires_mcp = [s.strip() for s in requires_mcp_val if s.strip()]
        else:
            raw = str(requires_mcp_val or "").strip()
            requires_mcp = (
                [s.strip() for s in raw.split(",") if s.strip()] if raw else []
            )

        state = _load_json(skill_dir / STATE_FILENAME)
        return SkillMeta(
            name=name,
            description=description,
            last_run_at=_parse_iso(state.get("last_run_at")),
            run_count=max(0, int(state.get("run_count", 0) or 0)),
            status=self._normalize_status(state.get("status")),
            next=_clip(state.get("next", ""), 100),
            requires_mcp=requires_mcp,
            builtin=builtin,
        )

    def scan_skills(self) -> list[SkillMeta]:
        """扫描全部 skill，按最近运行时间倒序。"""
        skills: list[SkillMeta] = []
        seen: set[str] = set()
        for root, builtin in self._skill_roots():
            if not root.exists():
                logger.info("[drift_state] skills 目录不存在: %s", root)
                continue
            for skill_dir in sorted(root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                if (
                    builtin
                    and self.builtin_skill_names
                    and skill_dir.name not in self.builtin_skill_names
                ):
                    continue
                skill = self._load_skill_meta(skill_dir, builtin=builtin)
                if skill is None:
                    continue
                if skill.name in seen:
                    logger.info(
                        "[drift_state] 跳过重复 skill=%s", skill.name
                    )
                    continue
                seen.add(skill.name)
                skills.append(skill)

        min_time = datetime.min.replace(tzinfo=timezone.utc)
        skills.sort(key=lambda s: s.last_run_at or min_time, reverse=True)
        return skills

    def valid_skill_names(self) -> set[str]:
        return {skill.name for skill in self.scan_skills()}

    def skill_dir_for(self, skill_name: str) -> Path | None:
        name = str(skill_name or "").strip()
        if not name:
            return None
        workspace_dir = self.skills_dir / name
        if (workspace_dir / SKILL_FILENAME).exists():
            return workspace_dir
        if self.include_builtin_skills and self.builtin_skills_dir is not None:
            builtin_dir = self.builtin_skills_dir / name
            if (builtin_dir / SKILL_FILENAME).exists():
                return builtin_dir
        return None

    # ------------------------------------------------------------------
    # 全局漂移历史
    # ------------------------------------------------------------------

    def load_drift(self) -> dict[str, Any]:
        """读取全局 drift.json（带清洗）。"""
        raw = _load_json(self.drift_file)
        recent_runs = raw.get("recent_runs")
        if not isinstance(recent_runs, list):
            recent_runs = []

        rows: list[dict[str, str]] = []
        for row in recent_runs:
            if not isinstance(row, dict):
                continue
            skill = _clip(row.get("skill", ""), 80)
            run_at = _clip(row.get("run_at", ""), 80)
            one_line = _clip(row.get("one_line", ""), 150)
            message_result = _clip(row.get("message_result", "silent"), 20)
            if message_result not in _VALID_MESSAGE_RESULTS:
                message_result = "silent"
            if not skill or not run_at or not one_line:
                continue
            rows.append({
                "skill": skill,
                "run_at": run_at,
                "one_line": one_line,
                "message_result": message_result,
            })
        return {
            "version": 1,
            "recent_runs": rows[-RECENT_RUNS_LIMIT:],
            "note": _clip(raw.get("note", ""), 150),
        }

    def save_finish(
        self,
        *,
        skill_used: str,
        one_line: str,
        next_action: str,
        message_result: str,
        note: str | None = None,
        now_utc: datetime | None = None,
    ) -> None:
        """记录一次 skill 运行：skill state.json + 全局 drift.json。"""
        now = now_utc or datetime.now(timezone.utc)
        skill_name = str(skill_used or "").strip()
        skill_dir = self.skill_dir_for(skill_name) or (self.skills_dir / skill_name)
        skill_dir.mkdir(parents=True, exist_ok=True)

        state = _load_json(skill_dir / STATE_FILENAME)
        _atomic_save_json(
            skill_dir / STATE_FILENAME,
            {
                "version": 1,
                "last_run_at": now.isoformat(),
                "run_count": max(0, int(state.get("run_count", 0) or 0)) + 1,
                "status": "in_progress",
                "next": _clip(next_action, 100),
            },
        )

        drift = self.load_drift()
        recent_runs = list(drift.get("recent_runs", []))
        recent_runs.append({
            "skill": skill_name,
            "run_at": now.isoformat(),
            "one_line": _clip(one_line, 150),
            "message_result": (
                message_result
                if message_result in _VALID_MESSAGE_RESULTS
                else "silent"
            ),
        })
        payload: dict[str, Any] = {
            "version": 1,
            "recent_runs": recent_runs[-RECENT_RUNS_LIMIT:],
            "note": drift.get("note", ""),
        }
        if note is not None:
            payload["note"] = _clip(note, 150)
        _atomic_save_json(self.drift_file, payload)
