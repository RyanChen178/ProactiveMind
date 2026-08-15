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
                agent.reset_session()
                print("（已新建会话，旧历史仍保留）\n")
                continue

            print("\nagent > ", end="", flush=True)
            async for chunk in agent.run_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(chat_repl())
