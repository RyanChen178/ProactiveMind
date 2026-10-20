"""主动链路漂移状态测试：frontmatter 解析 + skill 扫描 + 运行记录。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from initiative.proactive_drift_state import (
    DRIFT_FILENAME,
    SKILL_FILENAME,
    STATE_FILENAME,
    DriftStateStore,
    SkillMeta,
    _clip,
    _parse_iso,
    parse_frontmatter,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "demo skill",
    requires_mcp: str | None = None,
    state: dict | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f"description: {description}"]
    if requires_mcp is not None:
        lines.append(f"requires_mcp: {requires_mcp}")
    lines.append("---")
    lines.append("")
    lines.append("正文内容")
    (skill_dir / SKILL_FILENAME).write_text("\n".join(lines), encoding="utf-8")
    if state is not None:
        (skill_dir / STATE_FILENAME).write_text(
            json.dumps(state), encoding="utf-8"
        )
    return skill_dir


class ParseFrontmatterTest(unittest.TestCase):
    def test_scalar_values(self) -> None:
        md = "---\nname: foo\ndescription: bar baz\n---\n正文"
        meta = parse_frontmatter(md)
        self.assertEqual(meta["name"], "foo")
        self.assertEqual(meta["description"], "bar baz")

    def test_list_values(self) -> None:
        md = "---\nname: foo\nrequires_mcp:\n  - server1\n  - server2\n---\n"
        meta = parse_frontmatter(md)
        self.assertEqual(meta["requires_mcp"], ["server1", "server2"])

    def test_no_frontmatter(self) -> None:
        self.assertEqual(parse_frontmatter("just text"), {})

    def test_empty_value_no_list(self) -> None:
        md = "---\nname: foo\nempty_key:\nnext: bar\n---\n"
        meta = parse_frontmatter(md)
        self.assertEqual(meta["empty_key"], "")
        self.assertEqual(meta["next"], "bar")

    def test_quoted_values_stripped(self) -> None:
        md = "---\nname: \"quoted\"\n---\n"
        meta = parse_frontmatter(md)
        self.assertEqual(meta["name"], "quoted")


class ClipTest(unittest.TestCase):
    def test_within_limit(self) -> None:
        self.assertEqual(_clip("hello", 10), "hello")

    def test_truncated(self) -> None:
        self.assertEqual(len(_clip("x" * 50, 10)), 10)

    def test_none_and_whitespace(self) -> None:
        self.assertEqual(_clip(None, 10), "")
        self.assertEqual(_clip("  ab  ", 10), "ab")


class ParseIsoTest(unittest.TestCase):
    def test_valid(self) -> None:
        dt = _parse_iso("2026-01-01T12:00:00+00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_invalid(self) -> None:
        self.assertIsNone(_parse_iso("not a date"))
        self.assertIsNone(_parse_iso(""))
        self.assertIsNone(_parse_iso(None))


class ScanSkillsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _store(self, **kwargs) -> DriftStateStore:
        return DriftStateStore(self.root, **kwargs)

    def test_empty_dir(self) -> None:
        store = self._store()
        self.assertEqual(store.scan_skills(), [])

    def test_scans_valid_skill(self) -> None:
        _write_skill(self.root / "skills", "demo")
        skills = self._store().scan_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "demo")
        self.assertFalse(skills[0].builtin)

    def test_skips_dir_without_skill_md(self) -> None:
        (self.root / "skills" / "orphan").mkdir(parents=True)
        self.assertEqual(self._store().scan_skills(), [])

    def test_skips_name_mismatch(self) -> None:
        # frontmatter name 与目录名不一致 → 跳过
        skill_dir = self.root / "skills" / "dirA"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_FILENAME).write_text(
            "---\nname: nameB\ndescription: x\n---\n", encoding="utf-8"
        )
        self.assertEqual(self._store().scan_skills(), [])

    def test_skips_missing_description(self) -> None:
        skill_dir = self.root / "skills" / "nodesc"
        skill_dir.mkdir(parents=True)
        (skill_dir / SKILL_FILENAME).write_text(
            "---\nname: nodesc\n---\n", encoding="utf-8"
        )
        self.assertEqual(self._store().scan_skills(), [])

    def test_sorted_by_last_run_desc(self) -> None:
        now = datetime.now(timezone.utc)
        _write_skill(
            self.root / "skills", "older",
            state={"last_run_at": (now - timedelta(hours=2)).isoformat()},
        )
        _write_skill(
            self.root / "skills", "newer",
            state={"last_run_at": (now - timedelta(minutes=5)).isoformat()},
        )
        skills = self._store().scan_skills()
        self.assertEqual([s.name for s in skills], ["newer", "older"])

    def test_never_run_sorted_last(self) -> None:
        _write_skill(self.root / "skills", "never")
        _write_skill(
            self.root / "skills", "ran",
            state={"last_run_at": datetime.now(timezone.utc).isoformat()},
        )
        skills = self._store().scan_skills()
        self.assertEqual(skills[0].name, "ran")
        self.assertEqual(skills[1].name, "never")

    def test_state_fields_loaded(self) -> None:
        now = datetime.now(timezone.utc)
        _write_skill(
            self.root / "skills", "stated",
            state={
                "last_run_at": now.isoformat(),
                "run_count": 5,
                "status": "in_progress",
                "next": "继续分析",
            },
        )
        skills = self._store().scan_skills()
        s = skills[0]
        self.assertEqual(s.run_count, 5)
        self.assertEqual(s.status, "in_progress")
        self.assertEqual(s.next, "继续分析")

    def test_invalid_status_normalized(self) -> None:
        _write_skill(
            self.root / "skills", "weird",
            state={"status": "bogus_status"},
        )
        skills = self._store().scan_skills()
        self.assertEqual(skills[0].status, "idle")

    def test_requires_mcp_list(self) -> None:
        _write_skill(
            self.root / "skills", "mcpuser",
            requires_mcp=None,  # 占位，下面手写 frontmatter
        )
        # 手写带列表的 frontmatter
        skill_dir = self.root / "skills" / "mcpuser"
        (skill_dir / SKILL_FILENAME).write_text(
            "---\nname: mcpuser\ndescription: d\nrequires_mcp:\n  - srv1\n  - srv2\n---\n",
            encoding="utf-8",
        )
        skills = self._store().scan_skills()
        self.assertEqual(skills[0].requires_mcp, ["srv1", "srv2"])

    def test_requires_mcp_comma_string(self) -> None:
        _write_skill(
            self.root / "skills", "commas", requires_mcp="a, b"
        )
        skills = self._store().scan_skills()
        self.assertEqual(skills[0].requires_mcp, ["a", "b"])

    def test_valid_skill_names(self) -> None:
        _write_skill(self.root / "skills", "one")
        _write_skill(self.root / "skills", "two")
        self.assertEqual(
            self._store().valid_skill_names(), {"one", "two"}
        )


class BuiltinSkillsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.builtin_root = self.root / "builtin"
        self.builtin_root.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_builtin_excluded_by_default(self) -> None:
        _write_skill(self.builtin_root, "bskill")
        store = DriftStateStore(self.root)
        self.assertEqual(store.scan_skills(), [])

    def test_builtin_included_when_enabled(self) -> None:
        _write_skill(self.builtin_root, "bskill")
        store = DriftStateStore(
            self.root,
            builtin_skills_dir=self.builtin_root,
            include_builtin_skills=True,
        )
        skills = store.scan_skills()
        self.assertEqual(len(skills), 1)
        self.assertTrue(skills[0].builtin)

    def test_builtin_filtered_by_names(self) -> None:
        _write_skill(self.builtin_root, "allowed")
        _write_skill(self.builtin_root, "blocked")
        store = DriftStateStore(
            self.root,
            builtin_skills_dir=self.builtin_root,
            include_builtin_skills=True,
            builtin_skill_names={"allowed"},
        )
        names = {s.name for s in store.scan_skills()}
        self.assertEqual(names, {"allowed"})

    def test_workspace_wins_over_builtin_duplicate(self) -> None:
        _write_skill(self.builtin_root, "dup")
        _write_skill(self.root / "skills", "dup")
        store = DriftStateStore(
            self.root,
            builtin_skills_dir=self.builtin_root,
            include_builtin_skills=True,
        )
        skills = store.scan_skills()
        self.assertEqual(len(skills), 1)
        self.assertFalse(skills[0].builtin)

    def test_skill_dir_for_prefers_workspace(self) -> None:
        _write_skill(self.builtin_root, "dup")
        ws_dir = _write_skill(self.root / "skills", "dup")
        store = DriftStateStore(
            self.root,
            builtin_skills_dir=self.builtin_root,
            include_builtin_skills=True,
        )
        self.assertEqual(store.skill_dir_for("dup"), ws_dir)

    def test_skill_dir_for_falls_back_to_builtin(self) -> None:
        b_dir = _write_skill(self.builtin_root, "only_builtin")
        store = DriftStateStore(
            self.root,
            builtin_skills_dir=self.builtin_root,
            include_builtin_skills=True,
        )
        self.assertEqual(store.skill_dir_for("only_builtin"), b_dir)

    def test_skill_dir_for_unknown(self) -> None:
        store = DriftStateStore(self.root)
        self.assertIsNone(store.skill_dir_for("nope"))
        self.assertIsNone(store.skill_dir_for(""))


class LoadDriftTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = DriftStateStore(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty(self) -> None:
        drift = self.store.load_drift()
        self.assertEqual(drift["recent_runs"], [])
        self.assertEqual(drift["note"], "")

    def test_corrupted_file_returns_empty(self) -> None:
        (self.root / DRIFT_FILENAME).write_text("not json", encoding="utf-8")
        drift = self.store.load_drift()
        self.assertEqual(drift["recent_runs"], [])

    def test_rows_cleaned(self) -> None:
        raw = {
            "version": 1,
            "recent_runs": [
                {"skill": "a", "run_at": "t1", "one_line": "ok",
                 "message_result": "sent"},
                {"skill": "", "run_at": "t2", "one_line": "missing skill"},
                {"skill": "c", "run_at": "", "one_line": "missing time"},
                {"skill": "d", "run_at": "t4", "one_line": "",
                 "message_result": "sent"},
                {"skill": "e", "run_at": "t5", "one_line": "bad result",
                 "message_result": "bogus"},
                "not-a-dict",
            ],
            "note": "x" * 300,
        }
        (self.root / DRIFT_FILENAME).write_text(
            json.dumps(raw), encoding="utf-8"
        )
        drift = self.store.load_drift()
        rows = drift["recent_runs"]
        # 缺 skill / run_at / one_line 的行被丢弃，坏 message_result 归一
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["skill"], "a")
        self.assertEqual(rows[1]["skill"], "e")
        self.assertEqual(rows[1]["message_result"], "silent")
        # note 截断到 150
        self.assertEqual(len(drift["note"]), 150)


class SaveFinishTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = DriftStateStore(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_creates_state_and_drift(self) -> None:
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        self.store.save_finish(
            skill_used="demo",
            one_line="完成了一次审计",
            next_action="下次继续",
            message_result="sent",
            now_utc=now,
        )
        # skill state.json
        state_file = self.root / "skills" / "demo" / STATE_FILENAME
        self.assertTrue(state_file.exists())
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["run_count"], 1)
        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["next"], "下次继续")

        # 全局 drift.json
        drift = self.store.load_drift()
        self.assertEqual(len(drift["recent_runs"]), 1)
        row = drift["recent_runs"][0]
        self.assertEqual(row["skill"], "demo")
        self.assertEqual(row["one_line"], "完成了一次审计")
        self.assertEqual(row["message_result"], "sent")

    def test_increments_run_count(self) -> None:
        now = datetime.now(timezone.utc)
        for i in range(3):
            self.store.save_finish(
                skill_used="demo",
                one_line=f"run {i}",
                next_action="",
                message_result="silent",
                now_utc=now,
            )
        state_file = self.root / "skills" / "demo" / STATE_FILENAME
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["run_count"], 3)

    def test_recent_runs_rolls_to_limit(self) -> None:
        now = datetime.now(timezone.utc)
        for i in range(15):
            self.store.save_finish(
                skill_used="demo",
                one_line=f"run {i}",
                next_action="",
                message_result="silent",
                now_utc=now,
            )
        drift = self.store.load_drift()
        self.assertEqual(len(drift["recent_runs"]), 10)

    def test_note_updated_when_provided(self) -> None:
        now = datetime.now(timezone.utc)
        self.store.save_finish(
            skill_used="d", one_line="x", next_action="",
            message_result="silent", note="重要备注", now_utc=now,
        )
        drift = self.store.load_drift()
        self.assertEqual(drift["note"], "重要备注")

    def test_note_preserved_when_none(self) -> None:
        now = datetime.now(timezone.utc)
        self.store.save_finish(
            skill_used="d", one_line="x", next_action="",
            message_result="silent", note="keep me", now_utc=now,
        )
        self.store.save_finish(
            skill_used="d", one_line="y", next_action="",
            message_result="silent", note=None, now_utc=now,
        )
        drift = self.store.load_drift()
        self.assertEqual(drift["note"], "keep me")

    def test_long_fields_truncated(self) -> None:
        now = datetime.now(timezone.utc)
        self.store.save_finish(
            skill_used="d",
            one_line="y" * 300,
            next_action="n" * 300,
            message_result="silent",
            now_utc=now,
        )
        drift = self.store.load_drift()
        self.assertEqual(len(drift["recent_runs"][0]["one_line"]), 150)
        state = json.loads(
            (self.root / "skills" / "d" / STATE_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(state["next"]), 100)

    def test_bad_message_result_normalized(self) -> None:
        now = datetime.now(timezone.utc)
        self.store.save_finish(
            skill_used="d", one_line="x", next_action="",
            message_result="weird", now_utc=now,
        )
        drift = self.store.load_drift()
        self.assertEqual(drift["recent_runs"][0]["message_result"], "silent")


class SkillMetaDataclassTest(unittest.TestCase):
    def test_defaults(self) -> None:
        s = SkillMeta(name="n", description="d")
        self.assertEqual(s.status, "idle")
        self.assertEqual(s.run_count, 0)
        self.assertFalse(s.builtin)
        self.assertEqual(s.requires_mcp, [])


if __name__ == "__main__":
    unittest.main()
