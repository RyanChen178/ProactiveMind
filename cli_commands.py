"""CLI 命令路由 —— 用户在 REPL 中输入的命令解析与执行。

支持的命令：
  /help              显示帮助
  /clear 或 /reset   新建会话
  /pending           列出待归档记忆
  /promote           提升待归档记忆
  /skills            列出所有可用的后台 playbook
  /skill <name>      手动触发某个 playbook
  /memory            显示长期记忆片段
  /notes [tag]       列出笔记（按 tag 过滤）
  /search <query>    语义搜索长期记忆
  /stats             显示 turn 统计
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class CommandContext:
    """命令执行时所需的 Agent 句柄集合。"""

    agent: Any
    proactive_loop: Any = None
    skills_dir: Any = None  # Path-like，指向 PLAYBOOK 目录
    notes_store: Any = None
    memory_query: Callable[[str, int], list[tuple[str, float]]] | None = None
    output: Callable[[str], None] = field(default_factory=lambda: print)


@dataclass
class CommandResult:
    """命令执行结果。"""

    handled: bool = False
    consumed: bool = False  # True 表示命令已消费用户输入，不需要再走 agent
    message: str = ""


# 命令注册表：command_name -> handler(ctx, args) -> CommandResult
CommandHandler = Callable[[CommandContext, list[str]], Awaitable[CommandResult]]

_REGISTRY: dict[str, CommandHandler] = {}


def register(name: str) -> Callable[[CommandHandler], CommandHandler]:
    """装饰器：把命令处理函数注册到全局表。"""

    def decorator(func: CommandHandler) -> CommandHandler:
        _REGISTRY[name] = func
        return func

    return decorator


def known_commands() -> list[str]:
    """返回所有已注册命令名（按字典序）。"""
    return sorted(_REGISTRY.keys())


def parse_command(line: str) -> tuple[str, list[str]] | None:
    """从用户输入中解析命令名 + 参数列表。

    Returns:
        (command_name, args) 或 None 表示不是命令。
    """
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = line.split(maxsplit=1)
    cmd = parts[0].lstrip("/")
    rest = parts[1].split() if len(parts) > 1 else []
    return cmd, rest


async def dispatch(ctx: CommandContext, line: str) -> CommandResult:
    """解析并执行一行命令。"""
    parsed = parse_command(line)
    if parsed is None:
        return CommandResult(handled=False)
    cmd, args = parsed
    handler = _REGISTRY.get(cmd)
    if handler is None:
        ctx.output(f"未知命令: /{cmd}（输入 /help 查看可用命令）")
        return CommandResult(handled=True, consumed=True)
    try:
        return await handler(ctx, args)
    except Exception as exc:
        log.exception("命令 /%s 执行失败", cmd)
        ctx.output(f"命令执行出错: {exc}")
        return CommandResult(handled=True, consumed=True)


# ---------------------------------------------------------------------------
# 命令实现
# ---------------------------------------------------------------------------


@register("help")
async def _cmd_help(ctx: CommandContext, args: list[str]) -> CommandResult:
    lines = [
        "可用命令：",
        "  /help              显示此帮助",
        "  /clear 或 /reset   新建会话",
        "  /pending           列出待归档记忆",
        "  /promote           提升待归档记忆",
        "  /optimize          立即执行一轮记忆归档",
        "  /skills            列出后台 playbook",
        "  /skill <name>      手动触发某个 playbook",
        "  /memory            显示长期记忆片段",
        "  /search <query>    语义搜索长期记忆",
        "  /notes [tag]       列出笔记（按 tag 过滤）",
        "  /tools             列出全部可用工具",
        "  /mcp               查看 MCP server 状态",
        "  /stats             显示 turn 统计",
    ]
    ctx.output("\n".join(lines))
    return CommandResult(handled=True, consumed=True)


@register("clear")
@register("reset")
async def _cmd_reset(ctx: CommandContext, args: list[str]) -> CommandResult:
    ctx.agent.reset_session()
    ctx.output("（已新建会话，旧历史仍保留）")
    return CommandResult(handled=True, consumed=True)


@register("pending")
async def _cmd_pending(ctx: CommandContext, args: list[str]) -> CommandResult:
    facts = ctx.agent.get_pending_memories()
    if not facts:
        ctx.output("（没有待归档记忆）")
    else:
        ctx.output("（待归档记忆）")
        ctx.output("\n".join(f"- {f}" for f in facts))
    return CommandResult(handled=True, consumed=True)


@register("promote")
async def _cmd_promote(ctx: CommandContext, args: list[str]) -> CommandResult:
    facts = ctx.agent.promote_pending_memories()
    if not facts:
        ctx.output("（没有新的候选记忆可提升）")
    else:
        ctx.output(f"（已提升 {len(facts)} 条候选记忆）")
    return CommandResult(handled=True, consumed=True)


@register("optimize")
async def _cmd_optimize(ctx: CommandContext, args: list[str]) -> CommandResult:
    """立即执行一轮 PENDING.md 归档（质量过滤 + 去重后并入 MEMORY.md）。"""
    run_now = getattr(ctx.agent, "run_optimizer_now", None)
    if run_now is None:
        ctx.output("（优化器不可用）")
        return CommandResult(handled=True, consumed=True)
    archived = await run_now()
    if archived:
        ctx.output(f"（已归档 {archived} 条候选记忆到长期记忆）")
    else:
        ctx.output("（本轮没有可归档的候选记忆）")
    return CommandResult(handled=True, consumed=True)


@register("skills")
async def _cmd_skills(ctx: CommandContext, args: list[str]) -> CommandResult:
    from pathlib import Path

    skills_dir = ctx.skills_dir
    if skills_dir is None or not Path(skills_dir).exists():
        ctx.output("（未配置 playbook 目录）")
        return CommandResult(handled=True, consumed=True)

    from initiative.drift import PLAYBOOK_FILENAME, _extract_first_heading

    entries: list[tuple[str, str]] = []
    for child in sorted(Path(skills_dir).iterdir()):
        if not child.is_dir():
            continue
        playbook = child / PLAYBOOK_FILENAME
        if not playbook.exists():
            continue
        desc = _extract_first_heading(playbook.read_text(encoding="utf-8"))
        entries.append((child.name, desc))

    if not entries:
        ctx.output("（没有可用的 playbook）")
    else:
        ctx.output(f"共 {len(entries)} 个 playbook：")
        for name, desc in entries:
            ctx.output(f"  /skill {name}   — {desc}")
    return CommandResult(handled=True, consumed=True)


@register("skill")
async def _cmd_run_skill(ctx: CommandContext, args: list[str]) -> CommandResult:
    if not args:
        ctx.output("用法: /skill <name>")
        return CommandResult(handled=True, consumed=True)
    skill_name = args[0]

    from pathlib import Path

    from initiative.drift import PLAYBOOK_FILENAME, WanderLoop

    skills_dir = ctx.skills_dir
    if skills_dir is None:
        ctx.output("（未配置 playbook 目录）")
        return CommandResult(handled=True, consumed=True)

    skill_path = Path(skills_dir) / skill_name / PLAYBOOK_FILENAME
    if not skill_path.exists():
        ctx.output(f"找不到 playbook: {skill_name}")
        return CommandResult(handled=True, consumed=True)

    # 复用 WanderLoop 的执行逻辑
    loop = WanderLoop(ctx.agent._provider, skills_dir)
    from initiative.drift import PlaybookEntry

    entry = PlaybookEntry(name=skill_name, path=skill_path)
    result = await loop._execute_skill(entry)
    if result.action == "executed":
        ctx.output(f"[/{skill_name}]\n{result.summary}")
    else:
        ctx.output(f"skill 执行失败: {result.summary}")
    return CommandResult(handled=True, consumed=True)


@register("memory")
async def _cmd_memory(ctx: CommandContext, args: list[str]) -> CommandResult:
    text = ctx.agent._memory.read_all()
    facts = [
        line.lstrip("- ").strip()
        for line in text.splitlines()
        if line.startswith("- ") and line[2:].strip()
    ]
    if not facts:
        ctx.output("（长期记忆为空）")
    else:
        ctx.output(f"共 {len(facts)} 条长期记忆：")
        for f in facts[:20]:
            ctx.output(f"- {f}")
        if len(facts) > 20:
            ctx.output(f"... 还有 {len(facts) - 20} 条")
    return CommandResult(handled=True, consumed=True)


@register("search")
async def _cmd_search(ctx: CommandContext, args: list[str]) -> CommandResult:
    if not args:
        ctx.output("用法: /search <query>")
        return CommandResult(handled=True, consumed=True)
    query = " ".join(args)
    if ctx.memory_query is None:
        ctx.output("（未挂载语义检索）")
        return CommandResult(handled=True, consumed=True)
    results = ctx.memory_query(query, 5)
    if not results:
        ctx.output(f"未找到与 '{query}' 相关的事实")
    else:
        ctx.output(f"搜索 '{query}' 找到 {len(results)} 条：")
        for fact, score in results:
            ctx.output(f"- (score={score:.2f}) {fact}")
    return CommandResult(handled=True, consumed=True)


@register("notes")
async def _cmd_notes(ctx: CommandContext, args: list[str]) -> CommandResult:
    if ctx.notes_store is None:
        ctx.output("（笔记扩展未启用）")
        return CommandResult(handled=True, consumed=True)
    tag = args[0] if args else None
    notes = ctx.notes_store.list(tag=tag, limit=50)
    if not notes:
        msg = "当前没有笔记" if not tag else f"标签 {tag} 下没有笔记"
        ctx.output(msg)
    else:
        prefix = f"标签 {tag} 下的笔记：" if tag else f"共 {len(notes)} 条笔记："
        ctx.output(prefix)
        for n in notes:
            tags = ",".join(n.get("tags", []))
            ctx.output(f"[{n['id']}] {n['content']}" + (f" # {tags}" if tags else ""))
    return CommandResult(handled=True, consumed=True)


@register("tools")
async def _cmd_tools(ctx: CommandContext, args: list[str]) -> CommandResult:
    """列出 Agent 当前可用的全部工具。"""
    tools = getattr(ctx.agent, "_tools", None)
    if tools is None:
        ctx.output("（工具注册表不可用）")
        return CommandResult(handled=True, consumed=True)
    schemas = tools.get_schemas()
    if not schemas:
        ctx.output("（没有注册任何工具）")
        return CommandResult(handled=True, consumed=True)
    ctx.output(f"共 {len(schemas)} 个工具：")
    for s in schemas:
        func = s.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        source = "MCP" if name.startswith("mcp_") else "内置"
        ctx.output(f"  [{source}] {name} — {desc}")
    return CommandResult(handled=True, consumed=True)


@register("mcp")
async def _cmd_mcp(ctx: CommandContext, args: list[str]) -> CommandResult:
    """查看 MCP server 连接状态与工具清单。"""
    registry = getattr(ctx.agent, "_mcp_registry", None)
    if registry is None:
        ctx.output("（未配置 MCP server）")
        return CommandResult(handled=True, consumed=True)
    clients = getattr(registry, "_clients", {})
    if not clients:
        ctx.output("（未配置 MCP server）")
        return CommandResult(handled=True, consumed=True)
    connected = 0
    for name, client in clients.items():
        status = "已连接" if client.is_connected else "未连接"
        if client.is_connected:
            connected += 1
        ctx.output(f"  {name} [{status}]")
        for tool in client.tools:
            desc = getattr(tool, "description", "") or ""
            ctx.output(f"    - mcp_{name}__{tool.name}  {desc[:60]}")
    ctx.output(f"共 {len(clients)} 个 server，{connected} 个已连接")
    return CommandResult(handled=True, consumed=True)


@register("stats")
async def _cmd_stats(ctx: CommandContext, args: list[str]) -> CommandResult:
    summary = ctx.agent._stats.summary()
    if summary.get("total_turns", 0) == 0:
        ctx.output("（暂无统计）")
    else:
        ctx.output(
            f"总轮次: {summary.get('total_turns', 0)}\n"
            f"总 token: {summary.get('total_tokens', 0)}\n"
            f"平均延迟: {summary.get('avg_latency_ms', 0.0):.0f} ms"
        )
    return CommandResult(handled=True, consumed=True)