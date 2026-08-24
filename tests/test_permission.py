"""工具权限系统测试。"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from agent.permission import (
    DenyRule,
    ToolPermission,
    create_default_permission,
)
from agent.provider import ToolCall
from agent.tools import Tool, ToolRegistry


async def _noop(_: dict[str, Any]) -> str:
    return "executed"


class DenyRuleTest(unittest.TestCase):
    def test_matches_shell_command_with_pattern(self) -> None:
        rule = DenyRule(
            tool_name="shell",
            pattern=r"\brm\s+-rf?\s+[/~]",
            reason="危险删除",
        )
        self.assertTrue(rule.matches("shell", {"command": "rm -rf /"}))
        self.assertTrue(rule.matches("shell", {"command": "rm -rf ~"}))
        self.assertTrue(rule.matches("shell", {"command": "rm -rf /home"}))

    def test_does_not_match_safe_command(self) -> None:
        rule = DenyRule(
            tool_name="shell",
            pattern=r"\brm\s+-rf?\s+[/~]",
        )
        self.assertFalse(rule.matches("shell", {"command": "rm file.txt"}))
        self.assertFalse(rule.matches("shell", {"command": "ls -la"}))

    def test_does_not_match_different_tool(self) -> None:
        rule = DenyRule(
            tool_name="shell",
            pattern=r"\brm\b",
        )
        self.assertFalse(rule.matches("memorize", {"fact": "rm something"}))

    def test_case_insensitive(self) -> None:
        rule = DenyRule(
            tool_name="shell",
            pattern=r"\bmkfs\b",
        )
        self.assertTrue(rule.matches("shell", {"command": "MKFS /dev/sda"}))


class ToolPermissionTest(unittest.TestCase):
    def test_allows_when_no_rules(self) -> None:
        perm = ToolPermission()
        allowed, reason = perm.check("shell", {"command": "ls"})
        self.assertTrue(allowed)
        self.assertEqual(reason, "")

    def test_denies_matching_rule(self) -> None:
        perm = ToolPermission()
        perm.add_rule(DenyRule(
            tool_name="shell",
            pattern=r"\brm\b",
            reason="禁止删除",
        ))
        allowed, reason = perm.check("shell", {"command": "rm file"})
        self.assertFalse(allowed)
        self.assertEqual(reason, "禁止删除")

    def test_allows_non_matching_rule(self) -> None:
        perm = ToolPermission()
        perm.add_rule(DenyRule(
            tool_name="shell",
            pattern=r"\brm\b",
        ))
        allowed, _ = perm.check("shell", {"command": "ls"})
        self.assertTrue(allowed)

    def test_rules_property_returns_copy(self) -> None:
        perm = ToolPermission()
        perm.add_rule(DenyRule(tool_name="shell", pattern="test"))
        rules = perm.rules
        rules.clear()
        self.assertEqual(len(perm.rules), 1)


class DefaultPermissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.perm = create_default_permission()

    def test_denies_rm_rf_root(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "rm -rf /"})
        self.assertFalse(allowed)

    def test_denies_rm_rf_home(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "rm -rf ~"})
        self.assertFalse(allowed)

    def test_denies_mkfs(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "mkfs.ext4 /dev/sda"})
        self.assertFalse(allowed)

    def test_denies_dd_write(self) -> None:
        allowed, _ = self.perm.check(
            "shell", {"command": "dd if=/dev/zero of=/dev/sda"}
        )
        self.assertFalse(allowed)

    def test_denies_shutdown(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "shutdown -h now"})
        self.assertFalse(allowed)

    def test_denies_chmod_777(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "chmod 777 /"})
        self.assertFalse(allowed)

    def test_allows_safe_commands(self) -> None:
        allowed, _ = self.perm.check("shell", {"command": "ls -la"})
        self.assertTrue(allowed)
        allowed, _ = self.perm.check("shell", {"command": "echo hello"})
        self.assertTrue(allowed)
        allowed, _ = self.perm.check("shell", {"command": "cat file.txt"})
        self.assertTrue(allowed)

    def test_allows_other_tools(self) -> None:
        allowed, _ = self.perm.check("memorize", {"fact": "rm -rf /"})
        self.assertTrue(allowed)


class ToolRegistryPermissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_tool_returns_denial_message(self) -> None:
        perm = ToolPermission()
        perm.add_rule(DenyRule(
            tool_name="shell",
            pattern=r"\brm\b",
            reason="禁止删除",
        ))
        registry = ToolRegistry(permission=perm)
        registry.register(
            Tool(
                name="shell",
                description="test",
                parameters={"type": "object", "properties": {}},
                func=_noop,
            )
        )

        result = await registry.execute(
            ToolCall(id="1", name="shell", arguments={"command": "rm file"})
        )

        self.assertIn("权限拒绝", result)
        self.assertIn("禁止删除", result)

    async def test_allowed_tool_executes_normally(self) -> None:
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="shell",
                description="test",
                parameters={"type": "object", "properties": {}},
                func=_noop,
            )
        )

        result = await registry.execute(
            ToolCall(id="1", name="shell", arguments={"command": "ls"})
        )

        self.assertEqual(result, "executed")


if __name__ == "__main__":
    unittest.main()
