"""ProactiveMind 入口。

启动模式：
  python main.py          —— CLI 对话 REPL
  python main.py web      —— Web Chat（http://127.0.0.1:6322）
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

    proactive_loop = InitiativeLoop(
        presence,
        is_passive_busy=lambda: False,
    )
    return agent, bus, presence, proactive_loop


async def chat_repl() -> None:
    agent, bus, presence, proactive_loop = _build_agent()
    proactive_task = asyncio.create_task(proactive_loop.run())

    print("ProactiveMind — 输入消息开始对话，/pending 查看记忆，Ctrl+C 退出\n")
    try:
        while True:
            try:
                user_input = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue
            if user_input in ("/clear", "/reset"):
                agent.reset_session()
                print("（已新建会话，旧历史仍保留）\n")
                continue
            if user_input == "/pending":
                facts = agent.get_pending_memories()
                if not facts:
                    print("（没有待归档记忆）\n")
                else:
                    print("（待归档记忆）")
                    print("\n".join(f"- {fact}" for fact in facts))
                    print()
                continue
            if user_input == "/promote":
                facts = agent.promote_pending_memories()
                if not facts:
                    print("（没有新的候选记忆可提升）\n")
                else:
                    print(f"（已提升 {len(facts)} 条候选记忆）\n")
                continue

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

    agent, bus, presence, proactive_loop = _build_agent()

    cm = SocketHub()
    proactive_loop._push_callback = cm.broadcast

    proactive_task = asyncio.create_task(proactive_loop.run())
    app = create_app(agent, cm)

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


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if mode == "web":
        asyncio.run(web_server())
    else:
        asyncio.run(chat_repl())
