"""Agent 循环 —— ReAct（推理 + 行动）。"""

from __future__ import annotations

import asyncio
import json
import logging

from agent.config import Config
from agent.consolidation import MemoryConsolidator
from agent.memory import MemoryStore
from agent.provider import LLMProvider, LLMResponse
from agent.prompt import PromptBuilder
from agent.session import Session
from agent.session_store import SessionStore
from agent.tools import ToolRegistry, build_default_tools
from agent.permission import create_default_permission
from bus import EventBus, TurnCommitted
from proactive.presence import PresenceStore
from plugins.manager import PluginManager

log = logging.getLogger(__name__)


class AgentLoop:
    """ReAct 循环：接收用户输入 → 调用 LLM → 执行工具 → 返回回复。"""

    def __init__(
        self,
        config: Config,
        bus: EventBus | None = None,
        presence: PresenceStore | None = None,
    ) -> None:
        self._config = config
        self._provider = LLMProvider(config.llm)
        self._memory = MemoryStore(config.workspace)
        self._session_store = SessionStore(config.workspace / "sessions.db")
        self._session_id = self._session_store.get_or_create_active_session()
        self._session = self._load_session(self._session_id)
        self._tools = build_default_tools(
            self._memory, permission=create_default_permission()
        )
        self._consolidator = MemoryConsolidator(self._provider, self._memory)
        self._bus = bus or EventBus()
        self._presence = presence
        self._plugin_manager: PluginManager | None = None
        self._load_plugins()
        self._register_bus_handlers()
        self._refresh_system_prompt()

    async def run(self, user_input: str, max_steps: int = 10) -> str:
        """执行一轮对话：用户输入 → 可能多轮工具调用 → 最终回复。"""

        self._session.add_user(user_input)
        if self._presence is not None:
            self._presence.record_user_message()

        for _step in range(max_steps):
            messages = self._build_messages()
            response = await self._provider.chat(
                messages, tools=self._tools.get_schemas()
            )

            # 没有 tool_calls → 最终回复
            if not response.tool_calls:
                self._session.add_assistant(response.content)
                await self._emit_turn_committed(user_input, response.content)
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
        if self._presence is not None:
            self._presence.record_user_message()
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
                await self._emit_turn_committed(user_input, response.content)
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

    def _register_bus_handlers(self) -> None:
        """注册事件总线 handler。"""

        async def on_turn_committed(event: TurnCommitted) -> None:
            cfg = getattr(self._config, "consolidation", None)
            if cfg is not None and cfg.enabled:
                try:
                    await self._consolidator.consolidate(
                        event.user_input, event.assistant_reply
                    )
                except Exception as exc:
                    log.warning("记忆归档失败: %s", exc)

        self._bus.on("turn_committed", on_turn_committed)

    async def _emit_turn_committed(
        self, user_input: str, assistant_reply: str
    ) -> None:
        """发布 TurnCommitted 事件，触发后台记忆归档等副作用。"""

        await self._bus.enqueue(
            TurnCommitted(
                session_id=self._session_id,
                user_input=user_input,
                assistant_reply=assistant_reply,
            )
        )

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

    def _load_plugins(self) -> None:
        """加载配置中指定的插件目录，注册工具。"""
        plugins_dir = getattr(self._config, "plugins_dir", None)
        if plugins_dir is None or not plugins_dir.exists():
            return
        self._plugin_manager = PluginManager(plugins_dir)
        loaded = self._plugin_manager.load_all(self._tools)
        if loaded:
            log.info("已加载 %d 个插件", len(loaded))

    async def aclose(self) -> None:
        try:
            await self._bus.drain()
            if self._plugin_manager is not None:
                await self._plugin_manager.unload_all()
            await self._provider.aclose()
        finally:
            self._session_store.close()
