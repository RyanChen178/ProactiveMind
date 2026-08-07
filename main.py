"""ProactiveMind 入口。

启动一个 CLI 对话 REPL：
  python main.py
"""

from __future__ import annotations

import asyncio

from agent.config import load_config
from agent.loop import AgentLoop


async def chat_repl() -> None:
    try:
        config = load_config("config.toml")
    except FileNotFoundError as exc:
        print(f"配置错误: {exc}")
        return

    agent = AgentLoop(config)

    print("ProactiveMind — 输入消息开始对话，Ctrl+C 退出\n")
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
                agent._session.clear()
                print("（已清空对话历史）\n")
                continue

            reply = await agent.run(user_input)
            print(f"\nagent > {reply}\n")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(chat_repl())
