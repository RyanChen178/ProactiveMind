"""ProactiveMind 入口。

启动一个 CLI 对话 REPL：
  python main.py
"""

from __future__ import annotations

import asyncio
import logging

from agent.config import load_config
from agent.loop import AgentLoop
from bus import EventBus
from proactive.loop import ProactiveLoop
from proactive.presence import PresenceStore


async def chat_repl() -> None:
    try:
        config = load_config("config.toml")
    except FileNotFoundError as exc:
        print(f"配置错误: {exc}")
        return

    bus = EventBus()
    bus.start()
    presence = PresenceStore(config.workspace / "presence.db")
    agent = AgentLoop(config, bus=bus, presence=presence)

    proactive_loop = ProactiveLoop(
        presence,
        is_passive_busy=lambda: False,
    )
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


if __name__ == "__main__":
    asyncio.run(chat_repl())
