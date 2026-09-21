"""ProactiveMind 入口。

启动模式：
  python main.py             —— CLI 对话 REPL
  python main.py web         —— Web Chat（http://127.0.0.1:6322）
  python main.py telegram    —— Telegram Bot 渠道
  python main.py dashboard   —— Dashboard 调试入口
  python main.py control     —— Agent Control Protocol 服务端（127.0.0.1:6324）
  python main.py setup       —— 交互式配置向导（生成 config.toml）
  python main.py supervise MODE —— Supervisor 模式托管 gateway
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mind.config import load_config
from mind.loop import MindLoop
from events import EventHub
from initiative.loop import InitiativeLoop
from initiative.presence import PresenceStore


def _build_agent():
    """加载配置，构建 Agent + InitiativeLoop + EventHub。"""
    try:
        config = load_config("config.toml")
    except FileNotFoundError as exc:
        print(f"配置错误: {exc}")
        raise

    bus = EventHub()
    bus.start()
    presence = PresenceStore(config.workspace / "presence.db")
    agent = MindLoop(config, bus=bus, presence=presence)

    from initiative.data_sources import DataSourceManager

    proactive_loop = InitiativeLoop(
        presence,
        is_passive_busy=agent.is_busy,
        data_source_manager=DataSourceManager(),
    )
    return agent, bus, presence, proactive_loop


async def _setup_mcp_quietly(agent: MindLoop) -> None:
    """启动配置中的 MCP server；失败不阻断主流程。"""
    try:
        count = await agent.setup_mcp()
        if count:
            print(f"（已加载 {count} 个 MCP 工具）")
    except Exception as exc:
        print(f"（MCP 加载失败: {exc}）")


def _start_background(agent: MindLoop) -> None:
    """启动 Agent 后台任务（PENDING.md 定时归档）。"""
    agent.start_optimizer_loop()


async def chat_repl() -> None:
    agent, bus, presence, proactive_loop = _build_agent()
    await _setup_mcp_quietly(agent)
    _start_background(agent)
    proactive_task = asyncio.create_task(proactive_loop.run())

    print("ProactiveMind — 输入消息开始对话，/help 查看命令，Ctrl+C 退出\n")

    from cli_commands import CommandContext, dispatch
    from pathlib import Path as _P
    from extensions import notes as notes_ext

    # 构造 CLI 命令上下文（绑定 agent / skills_dir / notes_store）
    config = agent._config
    notes_store = notes_ext._NoteStore(config.workspace)

    def _memory_query(query: str, top_k: int) -> list[tuple[str, float]]:
        try:
            return agent._memory.semantic_recall(query, top_k=top_k)
        except Exception:
            return []

    ctx = CommandContext(
        agent=agent,
        proactive_loop=proactive_loop,
        skills_dir=_P(config.workspace).parent / "playbooks",
        notes_store=notes_store,
        memory_query=_memory_query,
        output=print,
    )

    try:
        while True:
            try:
                user_input = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue
            if user_input.startswith("/"):
                result = await dispatch(ctx, user_input)
                if result.consumed:
                    continue
                # 如果没消费，也走 agent
            print("\nagent > ", end="", flush=True)
            async for chunk in agent.run_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
    finally:
        proactive_loop.stop()
        proactive_task.cancel()
        try:
            await proactive_task
        except asyncio.CancelledError:
            pass
        await agent.aclose()
        await bus.aclose()
        presence.close()


async def web_server() -> None:
    """启动 Web Chat 服务。"""
    import uvicorn

    from gateways.web_chat import SocketHub, create_app
    from mind.health import create_health_checker

    agent, bus, presence, proactive_loop = _build_agent()
    await _setup_mcp_quietly(agent)
    _start_background(agent)

    cm = SocketHub()
    proactive_loop._push_callback = cm.broadcast

    hc = create_health_checker(
        memory_store=agent._memory,
        session_store=agent._session_store,
        presence_store=presence,
    )

    proactive_task = asyncio.create_task(proactive_loop.run())
    app = create_app(agent, cm, health_checker=hc)

    config = uvicorn.Config(app, host="127.0.0.1", port=6322, log_level="info")
    server = uvicorn.Server(config)

    print("ProactiveMind Web Chat — http://127.0.0.1:6322")
    try:
        await server.serve()
    finally:
        proactive_loop.stop()
        proactive_task.cancel()
        try:
            await proactive_task
        except asyncio.CancelledError:
            pass
        await agent.aclose()
        await bus.aclose()
        presence.close()


async def telegram_gateway() -> None:
    """启动 Telegram Bot 渠道。"""
    from gateways.telegram_bot import run_telegram_gateway

    config = load_config("config.toml")
    bus = EventHub()
    bus.start()
    presence = PresenceStore(config.workspace / "presence.db")
    agent = MindLoop(config, bus=bus, presence=presence)

    try:
        await _setup_mcp_quietly(agent)
        _start_background(agent)
        await run_telegram_gateway(agent, config)
    finally:
        await agent.aclose()
        await bus.aclose()
        presence.close()


def run_setup_cli(args: list[str]) -> None:
    """启动交互式 Setup 向导。"""
    from bootstrap.setup import run_setup_from_args

    parsed = _parse_setup_args(args)
    try:
        result = run_setup_from_args(parsed)
        print(f"配置已写入: {result}")
    except SetupError as exc:
        print(f"错误: {exc}")
        sys.exit(1)


def run_supervisor_cli(args: list[str]) -> None:
    """以 Supervisor 模式托管 gateway 子进程。"""
    from bootstrap.supervisor import supervise

    if not args:
        print("用法: python app.py supervise <web|telegram>")
        sys.exit(1)
    mode = args[0]
    if mode not in ("web", "telegram"):
        print(f"不支持的 gateway 模式: {mode}")
        sys.exit(1)
    try:
        config = load_config("config.toml")
    except FileNotFoundError as exc:
        print(f"配置错误: {exc}")
        sys.exit(1)
    workspace = config.workspace
    exit_code = supervise(mode, workspace)
    sys.exit(exit_code)


async def dashboard_server() -> None:
    """启动 Dashboard 调试入口（http://127.0.0.1:6323）。"""
    import uvicorn

    from gateways.dashboard import create_dashboard_app

    config = load_config("config.toml")
    bus = EventHub()
    bus.start()
    presence = PresenceStore(config.workspace / "presence.db")
    agent = MindLoop(config, bus=bus, presence=presence)

    app = create_dashboard_app(agent, presence, config.workspace)

    uconfig = uvicorn.Config(app, host="127.0.0.1", port=6323, log_level="info")
    server = uvicorn.Server(uconfig)

    print("ProactiveMind Dashboard — http://127.0.0.1:6323")
    try:
        await server.serve()
    finally:
        await agent.aclose()
        await bus.aclose()
        presence.close()


async def control_server_entry() -> None:
    """启动 Agent Control Protocol 服务端（默认 127.0.0.1:6324）。"""
    from gateways.control_server import ControlServer

    agent, bus, presence, _ = _build_agent()
    await _setup_mcp_quietly(agent)
    _start_background(agent)

    server = ControlServer(agent, host="127.0.0.1", port=6324)
    port = await server.start()
    print(f"ProactiveMind Control — 127.0.0.1:{port}（JSON-RPC over TCP）")
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
        await agent.aclose()
        await bus.aclose()
        presence.close()


def _parse_setup_args(args: list[str]) -> dict[str, str | bool]:
    """解析 setup 命令行参数为 dict。"""
    parsed: dict[str, str | bool] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--non-interactive":
            parsed["non_interactive"] = True
            i += 1
        elif token in ("--output", "-o") and i + 1 < len(args):
            parsed["output"] = args[i + 1]
            i += 2
        elif token == "--overwrite":
            parsed["overwrite"] = True
            i += 1
        elif token == "--provider" and i + 1 < len(args):
            parsed["provider"] = args[i + 1]
            i += 2
        elif token == "--model" and i + 1 < len(args):
            parsed["model"] = args[i + 1]
            i += 2
        elif token == "--api-key" and i + 1 < len(args):
            parsed["api_key"] = args[i + 1]
            i += 2
        elif token == "--base-url" and i + 1 < len(args):
            parsed["base_url"] = args[i + 1]
            i += 2
        elif token == "--workspace" and i + 1 < len(args):
            parsed["workspace"] = args[i + 1]
            i += 2
        else:
            print(f"忽略未知参数: {token}")
            i += 1
    return parsed


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if mode == "web":
        asyncio.run(web_server())
    elif mode == "telegram":
        asyncio.run(telegram_gateway())
    elif mode == "dashboard":
        asyncio.run(dashboard_server())
    elif mode == "control":
        asyncio.run(control_server_entry())
    elif mode == "setup":
        run_setup_cli(sys.argv[2:])
    elif mode == "supervise":
        run_supervisor_cli(sys.argv[2:])
    else:
        asyncio.run(chat_repl())
