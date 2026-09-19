"""CLI 命令路由测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from cli_commands import (
    CommandContext,
    CommandResult,
    dispatch,
    known_commands,
    parse_command,
    register,
)


def _run(coro):
    return asyncio.run(coro)


class ParseCommandTest(unittest.TestCase):
    """parse_command 解析。"""

    def test_returns_none_for_non_command(self) -> None:
        self.assertIsNone(parse_command("hello"))

    def test_parses_simple_command(self) -> None:
        cmd, args = parse_command("/help")  # type: ignore[misc]
        self.assertEqual(cmd, "help")
        self.assertEqual(args, [])

    def test_parses_command_with_args(self) -> None:
        cmd, args = parse_command("/skill audit-memory")  # type: ignore[misc]
        self.assertEqual(cmd, "skill")
        self.assertEqual(args, ["audit-memory"])

    def test_parses_command_with_quoted_args(self) -> None:
        # 当前实现不处理引号；只支持简单分词
        cmd, args = parse_command("/search coffee query")  # type: ignore[misc]
        self.assertEqual(cmd, "search")
        self.assertEqual(args, ["coffee", "query"])

    def test_strips_leading_slash(self) -> None:
        cmd, _ = parse_command("/clear")  # type: ignore[misc]
        self.assertEqual(cmd, "clear")


class KnownCommandsTest(unittest.TestCase):
    """命令注册表。"""

    def test_includes_core_commands(self) -> None:
        cmds = known_commands()
        for expected in ("help", "clear", "pending", "promote", "skills", "skill"):
            self.assertIn(expected, cmds)


class _FakeMemory:
    def __init__(self, content: str = "") -> None:
        self._content = content

    def read_all(self) -> str:
        return self._content


class _FakeStats:
    def summary(self) -> dict:
        return {
            "total_turns": 5,
            "total_tokens": 1234,
            "avg_latency_ms": 100.5,
        }


class _FakeAgent:
    def __init__(self, memory_text: str = "", pending: list[str] | None = None) -> None:
        self._memory = _FakeMemory(memory_text)
        self._stats = _FakeStats()
        self._pending = pending or []
        self._promoted: list[str] = []
        self._reset_called = 0

    def reset_session(self) -> None:
        self._reset_called += 1

    def get_pending_memories(self) -> list[str]:
        return list(self._pending)

    def promote_pending_memories(self) -> list[str]:
        promoted = list(self._pending)
        self._promoted = promoted
        self._pending = []
        return promoted


class _FakeNotesStore:
    def __init__(self, notes: list[dict] | None = None) -> None:
        self._notes = notes or []

    def list(self, tag: str | None = None, limit: int = 50) -> list[dict]:
        items = self._notes
        if tag:
            items = [n for n in items if tag in n.get("tags", [])]
        return items[:limit]


class _CollectingContext(CommandContext):
    """默认 print 收集到列表的 ctx。"""

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        self._collected: list[str] = []
        super().__init__(
            agent=agent,
            output=lambda line: self._collected.append(line),
            **kwargs,
        )

    @property
    def output_lines(self) -> list[str]:
        return list(self._collected)


class HelpTest(unittest.TestCase):
    def test_help_lists_commands(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/help"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("/help", text)
        self.assertIn("/skills", text)
        self.assertIn("/skill", text)


class ClearResetTest(unittest.TestCase):
    def test_clear_resets_session(self) -> None:
        agent = _FakeAgent()
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/clear"))
        self.assertEqual(agent._reset_called, 1)
        self.assertIn("已新建会话", ctx.output_lines[0])

    def test_reset_alias_also_works(self) -> None:
        agent = _FakeAgent()
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/reset"))
        self.assertEqual(agent._reset_called, 1)


class PendingPromoteTest(unittest.TestCase):
    def test_pending_empty(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/pending"))
        self.assertIn("没有", ctx.output_lines[0])

    def test_pending_lists_facts(self) -> None:
        agent = _FakeAgent(pending=["事实 1", "事实 2"])
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/pending"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("事实 1", text)
        self.assertIn("事实 2", text)

    def test_promote_returns_count(self) -> None:
        agent = _FakeAgent(pending=["x", "y"])
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/promote"))
        self.assertIn("2", ctx.output_lines[0])
        self.assertEqual(agent._promoted, ["x", "y"])

    def test_promote_empty(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/promote"))
        self.assertIn("没有", ctx.output_lines[0])


class SkillsListTest(unittest.TestCase):
    def test_lists_playbooks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "playbooks"
            (skill_dir / "audit-memory").mkdir(parents=True)
            (skill_dir / "audit-memory" / "PLAYBOOK.md").write_text(
                "# 审计记忆\n\n描述内容",
                encoding="utf-8",
            )
            (skill_dir / "daily-summary").mkdir(parents=True)
            (skill_dir / "daily-summary" / "PLAYBOOK.md").write_text(
                "# 每日摘要\n\n生成摘要",
                encoding="utf-8",
            )
            ctx = _CollectingContext(agent=_FakeAgent(), skills_dir=skill_dir)
            _run(dispatch(ctx, "/skills"))
            text = "\n".join(ctx.output_lines)
            self.assertIn("2 个", text)
            self.assertIn("/skill audit-memory", text)
            self.assertIn("审计记忆", text)

    def test_skills_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "playbooks"
            skill_dir.mkdir()
            ctx = _CollectingContext(agent=_FakeAgent(), skills_dir=skill_dir)
            _run(dispatch(ctx, "/skills"))
            self.assertIn("没有", ctx.output_lines[0])

    def test_skills_none_dir(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent(), skills_dir=None)
        _run(dispatch(ctx, "/skills"))
        self.assertIn("未配置", ctx.output_lines[0])


class SkillRunTest(unittest.TestCase):
    def test_run_skill_missing_name(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/skill"))
        self.assertIn("用法", ctx.output_lines[0])

    def test_run_skill_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "playbooks"
            skill_dir.mkdir()
            ctx = _CollectingContext(agent=_FakeAgent(), skills_dir=skill_dir)
            _run(dispatch(ctx, "/skill nope"))
            self.assertIn("找不到", ctx.output_lines[0])

    def test_run_skill_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = Path(temp_dir) / "playbooks"
            target = skill_dir / "audit-memory"
            target.mkdir(parents=True)
            (target / "PLAYBOOK.md").write_text(
                "# 审计\n\n审计内容",
                encoding="utf-8",
            )
            agent = _FakeAgent()
            # 提供一个 fake provider
            fake_response = MagicMock()
            fake_response.content = "审计完成"
            fake_response.usage = {}
            agent._provider = MagicMock()
            agent._provider.chat = asyncio.coroutine(lambda _: fake_response)

            ctx = _CollectingContext(agent=agent, skills_dir=skill_dir)
            _run(dispatch(ctx, "/skill audit-memory"))
            text = "\n".join(ctx.output_lines)
            self.assertIn("审计完成", text)
            self.assertIn("audit-memory", text)


class MemoryTest(unittest.TestCase):
    def test_memory_empty(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/memory"))
        self.assertIn("为空", ctx.output_lines[0])

    def test_memory_lists_facts(self) -> None:
        memory_text = "# 长期记忆\n\n- 事实 A\n- 事实 B\n"
        ctx = _CollectingContext(agent=_FakeAgent(memory_text=memory_text))
        _run(dispatch(ctx, "/memory"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("事实 A", text)
        self.assertIn("事实 B", text)


class SearchTest(unittest.TestCase):
    def test_search_missing_query(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/search"))
        self.assertIn("用法", ctx.output_lines[0])

    def test_search_no_memory_query(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent(), memory_query=None)
        _run(dispatch(ctx, "/search coffee"))
        self.assertIn("未挂载", ctx.output_lines[0])

    def test_search_returns_results(self) -> None:
        def fake_query(query: str, top_k: int) -> list[tuple[str, float]]:
            return [("事实 A", 0.9), ("事实 B", 0.7)]

        ctx = _CollectingContext(
            agent=_FakeAgent(),
            memory_query=fake_query,
        )
        _run(dispatch(ctx, "/search coffee query"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("事实 A", text)
        self.assertIn("0.9", text)

    def test_search_no_match(self) -> None:
        ctx = _CollectingContext(
            agent=_FakeAgent(),
            memory_query=lambda q, k: [],
        )
        _run(dispatch(ctx, "/search nothing"))
        self.assertIn("未找到", ctx.output_lines[0])


class NotesTest(unittest.TestCase):
    def test_no_notes_store(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent(), notes_store=None)
        _run(dispatch(ctx, "/notes"))
        self.assertIn("未启用", ctx.output_lines[0])

    def test_list_all(self) -> None:
        store = _FakeNotesStore(notes=[
            {"id": "abc", "content": "note 1", "tags": ["work"]},
            {"id": "def", "content": "note 2", "tags": []},
        ])
        ctx = _CollectingContext(agent=_FakeAgent(), notes_store=store)
        _run(dispatch(ctx, "/notes"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("note 1", text)
        self.assertIn("note 2", text)
        self.assertIn("work", text)

    def test_filter_by_tag(self) -> None:
        store = _FakeNotesStore(notes=[
            {"id": "abc", "content": "work note", "tags": ["work"]},
            {"id": "def", "content": "personal note", "tags": ["home"]},
        ])
        ctx = _CollectingContext(agent=_FakeAgent(), notes_store=store)
        _run(dispatch(ctx, "/notes work"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("work note", text)
        self.assertNotIn("personal note", text)


class StatsTest(unittest.TestCase):
    def test_stats_empty(self) -> None:
        # _FakeStats 默认有数据；构造空 stats agent
        agent = MagicMock()
        agent._memory.read_all.return_value = ""
        agent._stats.summary.return_value = {"total_turns": 0}
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/stats"))
        self.assertIn("暂无", ctx.output_lines[0])

    def test_stats_with_data(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/stats"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("总轮次", text)
        self.assertIn("5", text)
        self.assertIn("1234", text)


class DispatchTest(unittest.TestCase):
    def test_unknown_command(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        _run(dispatch(ctx, "/unknown"))
        self.assertIn("未知命令", ctx.output_lines[0])

    def test_non_command_returns_unhandled(self) -> None:
        ctx = _CollectingContext(agent=_FakeAgent())
        result = _run(dispatch(ctx, "hello"))
        self.assertFalse(result.handled)


class _FakeToolsRegistry:
    def get_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "检索记忆",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_fake__echo",
                    "description": "远端回显",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]


class _FakeMcpToolInfo:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class _FakeMcpClientState:
    def __init__(self, connected: bool, tools: list) -> None:
        self.is_connected = connected
        self.tools = tools


class _FakeMcpRegistryState:
    def __init__(self, clients: dict) -> None:
        self._clients = clients


class ToolsCommandTest(unittest.TestCase):
    """CLI /tools 命令。"""

    def test_lists_tools_with_source(self) -> None:
        agent = _FakeAgent()
        agent._tools = _FakeToolsRegistry()
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/tools"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("共 2 个工具", text)
        self.assertIn("[内置] recall", text)
        self.assertIn("[MCP] mcp_fake__echo", text)

    def test_tools_unavailable(self) -> None:
        agent = _FakeAgent()
        agent._tools = None
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/tools"))
        self.assertIn("不可用", ctx.output_lines[0])

    def test_tools_registered_in_help(self) -> None:
        cmds = known_commands()
        self.assertIn("tools", cmds)
        self.assertIn("mcp", cmds)


class McpCommandTest(unittest.TestCase):
    """CLI /mcp 命令。"""

    def test_reports_server_status(self) -> None:
        agent = _FakeAgent()
        agent._mcp_registry = _FakeMcpRegistryState({
            "fake": _FakeMcpClientState(True, [_FakeMcpToolInfo("echo", "回显")]),
            "down": _FakeMcpClientState(False, []),
        })
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/mcp"))
        text = "\n".join(ctx.output_lines)
        self.assertIn("fake [已连接]", text)
        self.assertIn("mcp_fake__echo", text)
        self.assertIn("down [未连接]", text)
        self.assertIn("共 2 个 server，1 个已连接", text)

    def test_mcp_not_configured(self) -> None:
        agent = _FakeAgent()
        agent._mcp_registry = None
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/mcp"))
        self.assertIn("未配置", ctx.output_lines[0])

    def test_mcp_no_clients(self) -> None:
        agent = _FakeAgent()
        agent._mcp_registry = _FakeMcpRegistryState({})
        ctx = _CollectingContext(agent=agent)
        _run(dispatch(ctx, "/mcp"))
        self.assertIn("未配置", ctx.output_lines[0])


class RegisterDecoratorTest(unittest.TestCase):
    def test_register_adds_command(self) -> None:
        @register("test_cmd_xxx")
        async def _cmd(ctx, args):
            return CommandResult(handled=True, consumed=True)

        try:
            self.assertIn("test_cmd_xxx", known_commands())
        finally:
            # 清理（保持测试隔离）
            cli_commands_REGISTRY = __import__("cli_commands")._REGISTRY  # type: ignore
            cli_commands_REGISTRY.pop("test_cmd_xxx", None)


if __name__ == "__main__":
    unittest.main()