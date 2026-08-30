"""Agent 循环 —— ReAct（推理 + 行动）。"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from mind.config import Config
from mind.consolidation import MemoryConsolidator
from mind.memory import MemoryStore
from mind.provider import LLMProvider, LLMResponse
from mind.prompt import PromptBuilder
from mind.session import Session
from mind.session_store import SessionStore
from mind.stats import TurnStats
from mind.tools import ToolRegistry, build_core_tools
from mind.permission import create_default_permission
from mind.vector_store import VectorStore
from mind.compaction import ContextCompactor
from events import EventHub, TurnCompleted
from initiative.presence import PresenceStore
from extensions.manager import ExtensionManager

log = logging.getLogger(__name__)


def _merge_usage(target: dict[str, int], source: dict[str, int]) -> None:
    """合并 LLM 返回的 token 用量到累积 dict。"""
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = source.get(key, 0)
        if isinstance(val, (int, float)):
            target[key] = target.get(key, 0) + int(val)


class MindLoop:
    """ReAct 循环：接收用户输入 → 调用 LLM → 执行工具 → 返回回复。"""

    def __init__(
        self,
        config: Config,
        bus: EventHub | None = None,
        presence: PresenceStore | None = None,
    ) -> None:
        self._config = config
        self._provider = LLMProvider.from_config(config)
        self._memory = MemoryStore(config.workspace)
        self._session_store = SessionStore(config.workspace / "sessions.db")
        self._session_id = self._session_store.get_or_create_active_session()
        self._session = self._load_session(self._session_id)
        self._vector_store = VectorStore()
        self._refresh_vector_store()
        self._tools = build_core_tools(
            self._memory,
            permission=create_default_permission(),
            vector_store=self._vector_store,
        )
        self._consolidator = MemoryConsolidator(self._provider, self._memory)
        # 初始化上下文压缩器
        context_window = getattr(self._config.llm, 'context_window', 128000)
        context_compaction = getattr(self._config, "context_compaction", None)
        keep_recent = getattr(context_compaction, "keep_recent_tokens", 20000) if context_compaction else 20000
        self._compactor = ContextCompactor(
            provider=self._provider,
            context_window=context_window,
            keep_recent_tokens=keep_recent,
        )
        self._bus = bus or EventHub()
        self._presence = presence
        self._extension_manager: ExtensionManager | None = None
        self._stats = TurnStats()
        self._load_extensions()
        self._register_bus_handlers()
        self._refresh_system_prompt()

    @property
    def stats(self) -> TurnStats:
        return self._stats

    async def run(self, user_input: str, max_steps: int = 10) -> str:
        """执行一轮对话：用户输入 → 可能多轮工具调用 → 最终回复。"""

        self._session.add_user(user_input)
        if self._presence is not None:
            self._presence.record_user_message()

        start = time.monotonic()
        total_usage: dict[str, int] = {}
        tool_names: list[str] = []

        for _step in range(max_steps):
            messages = self._build_messages()
            response = await self._provider.chat(
                messages, tools=self._tools.get_schemas()
            )
            _merge_usage(total_usage, response.usage)

            # 没有 tool_calls → 最终回复
            if not response.tool_calls:
                self._session.add_assistant(response.content)
                await self._emit_turn_committed(user_input, response.content)
                latency_ms = (time.monotonic() - start) * 1000
                self._stats.record(
                    session_id=self._session_id,
                    user_input=user_input,
                    assistant_reply=response.content,
                    tool_calls=tool_names,
                    usage=total_usage,
                    latency_ms=latency_ms,
                )
                return response.content

            # 有 tool_calls → 执行工具后继续循环
            tool_names.extend(c.name for c in response.tool_calls)
            await self._execute_tool_calls(response)

        message = "（达到最大工具调用次数，终止本轮）"
        self._session.add_assistant(message)
        latency_ms = (time.monotonic() - start) * 1000
        self._stats.record(
            session_id=self._session_id,
            user_input=user_input,
            assistant_reply=message,
            tool_calls=tool_names,
            usage=total_usage,
            latency_ms=latency_ms,
        )
        return message

    async def run_stream(
        self, user_input: str, max_steps: int = 10
    ):
        """执行一轮对话，并逐段产出最终回复文本。"""

        self._session.add_user(user_input)
        if self._presence is not None:
            self._presence.record_user_message()

        start = time.monotonic()
        total_usage: dict[str, int] = {}
        tool_names: list[str] = []

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
            _merge_usage(total_usage, response.usage)
            if not response.tool_calls:
                self._session.add_assistant(response.content)
                await self._emit_turn_committed(user_input, response.content)
                latency_ms = (time.monotonic() - start) * 1000
                self._stats.record(
                    session_id=self._session_id,
                    user_input=user_input,
                    assistant_reply=response.content,
                    tool_calls=tool_names,
                    usage=total_usage,
                    latency_ms=latency_ms,
                )
                return
            tool_names.extend(c.name for c in response.tool_calls)
            await self._execute_tool_calls(response)

        message = "（达到最大工具调用次数，终止本轮）"
        self._session.add_assistant(message)
        latency_ms = (time.monotonic() - start) * 1000
        self._stats.record(
            session_id=self._session_id,
            user_input=user_input,
            assistant_reply=message,
            tool_calls=tool_names,
            usage=total_usage,
            latency_ms=latency_ms,
        )
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
        
        # 检查是否需要上下文压缩
        if self._compactor.should_compact(messages):
            log.info("触发上下文压缩")
            try:
                messages, checkpoint = self._compactor.compact(messages)
                log.info(
                    "压缩完成: generation=%d, tokens=%d→%d",
                    checkpoint.generation,
                    checkpoint.estimated_tokens_before,
                    checkpoint.estimated_tokens_after,
                )
            except Exception as exc:
                log.warning("上下文压缩失败: %s", exc)
        
        return messages


    def _register_bus_handlers(self) -> None:
        """注册事件总线 handler。"""

        async def on_turn_committed(event: TurnCompleted) -> None:
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
        """发布 TurnCompleted 事件，触发后台记忆归档等副作用。"""

        await self._bus.enqueue(
            TurnCompleted(
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

    def list_sessions(self) -> list[dict]:
        """列出所有会话。"""
        return self._session_store.list_sessions()

    def get_session_history(self, session_id: str) -> list[dict] | None:
        """获取指定会话的消息历史。"""
        info = self._session_store.get_session_id_for_export(session_id)
        if info is None:
            return None
        return self._session_store.load_messages(session_id)

    def switch_session(self, session_id: str) -> bool:
        """切换到已有会话，返回是否成功。"""
        info = self._session_store.get_session_id_for_export(session_id)
        if info is None:
            return False
        self._session_id = session_id
        self._session = self._load_session(session_id)
        return True

    def export_session_markdown(self, session_id: str) -> str | None:
        """将会话历史导出为 Markdown。"""
        info = self._session_store.get_session_id_for_export(session_id)
        if info is None:
            return None
        messages = self._session_store.load_messages(session_id)
        lines = [f"# 会话导出 {session_id}", ""]
        lines.append(f"创建时间：{info['created_at']}")
        lines.append("")
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"## 用户")
                lines.append("")
                lines.append(content)
                lines.append("")
            elif role == "assistant":
                lines.append(f"## 助手")
                lines.append("")
                lines.append(content)
                lines.append("")
            elif role == "tool":
                lines.append(f"<details><summary>工具结果 ({msg.get('tool_call_id', '')})</summary>")
                lines.append("")
                lines.append(f"```\n{content}\n```")
                lines.append("")
                lines.append("</details>")
                lines.append("")
        return "\n".join(lines)

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
        self._refresh_vector_store()

    def _refresh_vector_store(self) -> None:
        """用当前长期记忆重建向量索引。"""
        all_text = self._memory.read_all().strip()
        facts = [
            line.lstrip("- ").strip()
            for line in all_text.splitlines()
            if line.strip().startswith("- ")
        ]
        if facts:
            self._vector_store.rebuild(facts)

    def _load_extensions(self) -> None:
        """加载配置中指定的插件目录，注册工具。"""
        extensions_dir = getattr(self._config, "extensions_dir", None)
        if extensions_dir is None or not extensions_dir.exists():
            return
        self._extension_manager = ExtensionManager(extensions_dir)
        loaded = self._extension_manager.load_all(self._tools)
        if loaded:
            log.info("已加载 %d 个插件", len(loaded))

    async def aclose(self) -> None:
        try:
            await self._bus.drain()
            if self._extension_manager is not None:
                await self._extension_manager.unload_all()
            await self._provider.aclose()
        finally:
            self._session_store.close()
