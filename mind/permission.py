"""工具权限系统 —— 执行前安全审查。

对工具调用做准入检查，默认拦截危险 shell 命令。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DenyRule:
    """一条拒绝规则。"""

    tool_name: str
    pattern: str
    reason: str = ""
    _compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name != self.tool_name:
            return False
        text = " ".join(str(v) for v in arguments.values())
        return bool(self._compiled.search(text))


class ToolPermission:
    """工具权限管理器。"""

    def __init__(self) -> None:
        self._rules: list[DenyRule] = []

    def add_rule(self, rule: DenyRule) -> None:
        self._rules.append(rule)

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """检查工具调用是否被允许。

        返回 (allowed, reason)。
        allowed=True 表示通过，reason 为空。
        allowed=False 表示拒绝，reason 为拒绝原因。
        """
        for rule in self._rules:
            if rule.matches(tool_name, arguments):
                reason = rule.reason or f"命令匹配拒绝规则: {rule.pattern}"
                log.warning("工具 %s 被拒绝: %s", tool_name, reason)
                return False, reason
        return True, ""

    @property
    def rules(self) -> list[DenyRule]:
        return list(self._rules)


def create_default_permission() -> ToolPermission:
    """创建默认权限规则——拦截常见危险 shell 命令。"""
    perm = ToolPermission()

    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r"\brm\s+-rf?\s+[/~]",
        reason="拒绝递归删除根目录或家目录",
    ))
    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r"\bmkfs\b",
        reason="拒绝格式化文件系统",
    ))
    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r"\bdd\b.*\bif=",
        reason="拒绝 dd 写入操作",
    ))
    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r":\(\)\s*\{.*\}",
        reason="拒绝 fork 炸弹",
    ))
    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r"\b(shutdown|reboot|halt|poweroff)\b",
        reason="拒绝关机/重启命令",
    ))
    perm.add_rule(DenyRule(
        tool_name="shell",
        pattern=r"\bchmod\s+777\b",
        reason="拒绝全局可写权限",
    ))

    return perm
