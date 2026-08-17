"""Agent 循环 —— ReAct（推理 + 行动）。"""

from __future__ import annotations

import asyncio
import json

from agent.config import Config
from agent.consolidation import MemoryConsolidator
from agent.memory import MemoryStore
from agent.provider import LLMProvider, LLMResponse
from agent.prompt import PromptBuilder
from agent.session import Session
from agent.session_store import SessionStore
from agent.tools import ToolRegistry, build_default_tools

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
        self._consolidator = MemoryConsolidator(self._provider, self._memory)
        self._consolidation_tasks: set[asyncio.Task[list[str]]] = set()
        self._refresh_system_prompt()

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
                self._schedule_consolidation(user_input, response.content)
                return response.content

            # 有 tool_calls → 执行工具后继续循环
            await self._execute_tool_calls(response)

        message = "（达到最大工具调用次数，终止本轮）"
        self._session.add_assistant(message)
        return message

    async def run_stream(
        self, user_input: str, max_steps: int = 10
    ):
        """执行一轮对话，并逐段产出最终回复文本。"""

        self._session.add_user(user_input)
        for _step in range(max_steps):
            response: LLMResponse | None = None
            async for event in self._provider.chat_stream(
                self._build_messages(), tools=self._tools.get_schemas()
            ):
                if event.content:
                    yield event.content
                if event.response is not None:
                    response = event.response
            if response is None:
                raise RuntimeError("LLM 流式响应缺少最终结果")
            if not response.tool_calls:
                self._session.add_assistant(response.content)
                self._schedule_consolidation(user_input, response.content)
                return
            await self._execute_tool_calls(response)

        message = "（达到最大工具调用次数，终止本轮）"
        self._session.add_assistant(message)
        yield message

    async def _execute_tool_calls(self, response: LLMResponse) -> None:
        """持久化模型工具调用，并按顺序执行工具。"""

        tool_call_dicts = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
        self._session.add_assistant(response.content, tool_calls=tool_call_dicts)
        for call in response.tool_calls:
            result = await self._tools.execute(call)
            self._session.add_tool_result(call.id, result)

    def _build_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(
            self._session.get_history(self._config.max_history_tokens)
        )
        return messages

    def _schedule_consolidation(
        self, user_input: str, assistant_reply: str
    ) -> None:
        """在回复已写入后后台提取候选长期记忆。"""

        config = getattr(self._config, "consolidation", None)
        if config is None or not config.enabled:
            return
        task = asyncio.create_task(
            self._consolidator.consolidate(user_input, assistant_reply)
        )
        self._consolidation_tasks.add(task)
        task.add_done_callback(self._observe_consolidation)

    def _observe_consolidation(self, task: asyncio.Task[list[str]]) -> None:
        """回收后台归档任务，并显式报告失败。"""

        self._consolidation_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"\n（记忆归档失败：{exc}）")

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

    def get_pending_memories(self) -> list[str]:
        """返回尚未提升的候选长期记忆。"""

        return self._memory.unpromoted_pending()

    def promote_pending_memories(self) -> list[str]:
        """将候选记忆提升到长期记忆。"""

        facts = self._memory.promote_pending()
        if facts:
            self._refresh_system_prompt()
        return facts

    def _refresh_system_prompt(self) -> None:
        """用最新长期记忆重建系统提示词。"""

        memory_text = self._memory.read_all().strip()
        self._system_prompt = PromptBuilder(self._config.prompt).build(memory_text)

    async def aclose(self) -> None:
        try:
            tasks = list(self._consolidation_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await self._provider.aclose()
        finally:
            self._session_store.close()
