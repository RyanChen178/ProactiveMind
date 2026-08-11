"""Agent 循环 —— ReAct（推理 + 行动）。"""

from __future__ import annotations

import json

from agent.config import Config
from agent.memory import MemoryStore
from agent.provider import LLMProvider, LLMResponse
from agent.session import Session
from agent.session_store import SessionStore
from agent.tools import ToolRegistry, build_default_tools

SYSTEM_PROMPT = """\
你是 ProactiveMind，一个有持久记忆的 AI 助手。

你可以使用工具来帮助用户：
- get_time: 获取当前时间
- shell: 执行 shell 命令
- memorize: 将重要事实保存到长期记忆
- recall: 从记忆中检索信息

使用工具时先思考是否真的需要，避免不必要的调用。"""


class AgentLoop:
    """ReAct 循环：接收用户输入 → 调用 LLM → 执行工具 → 返回回复。"""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._provider = LLMProvider(config.llm)
        self._memory = MemoryStore(config.workspace)
        self._session_store = SessionStore(config.workspace / "sessions.db")
        self._session_id = self._session_store.get_or_create_active_session()
        self._session = self._load_session(self._session_id)
        self._tools = build_default_tools(self._memory)

        # 把已有记忆注入 system prompt
        memory_text = self._memory.read_all().strip()
        if memory_text and len(memory_text) > 50:
            self._system_prompt = SYSTEM_PROMPT + f"\n\n## 已有记忆\n{memory_text}"
        else:
            self._system_prompt = SYSTEM_PROMPT

    async def run(self, user_input: str, max_steps: int = 10) -> str:
        """执行一轮对话：用户输入 → 可能多轮工具调用 → 最终回复。"""

        self._session.add_user(user_input)

        for _step in range(max_steps):
            messages = self._build_messages()
            response = await self._provider.chat(
                messages, tools=self._tools.get_schemas()
            )

            # 没有 tool_calls → 最终回复
            if not response.tool_calls:
                self._session.add_assistant(response.content)
                return response.content

            # 有 tool_calls → 执行工具后继续循环
            tool_call_dicts = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments, ensure_ascii=False
                        ),
                    },
                }
                for call in response.tool_calls
            ]
            self._session.add_assistant(
                response.content, tool_calls=tool_call_dicts
            )

            for call in response.tool_calls:
                result = await self._tools.execute(call)
                self._session.add_tool_result(call.id, result)

        message = "（达到最大工具调用次数，终止本轮）"
        self._session.add_assistant(message)
        return message

    def _build_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(
            self._session.get_history(self._config.max_history_tokens)
        )
        return messages

    def _load_session(self, session_id: str) -> Session:
        """从 SQLite 恢复指定会话的内存视图。"""

        messages = self._session_store.load_messages(session_id)
        return Session(
            messages,
            persist_message=lambda message: self._session_store.append_message(
                session_id, message
            ),
        )

    def reset_session(self) -> None:
        """切换到新的活动会话，保留旧会话历史。"""

        self._session_id = self._session_store.create_active_session()
        self._session = self._load_session(self._session_id)

    async def aclose(self) -> None:
        try:
            await self._provider.aclose()
        finally:
            self._session_store.close()
