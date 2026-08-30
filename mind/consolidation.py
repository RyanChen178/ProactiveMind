"""对话候选记忆提取与待归档缓冲。"""

from __future__ import annotations

import json
import logging

from mind.memory import MemoryStore
from mind.provider import LLMProvider

log = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """\
你负责从一轮用户与助手对话中提取值得长期保留的用户事实。
只保留稳定的偏好、身份背景、长期项目目标、明确承诺或可复用流程；
不要记录临时问题、助手猜测、工具输出、敏感信息或没有事实价值的内容。
只输出 JSON：{"facts":["事实一","事实二"]}。没有可保留事实时输出 {"facts":[]}。"""


class MemoryConsolidator:
    """将模型提取的候选事实写入 PENDING.md。"""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryStore,
        lightweight_assistant: "LightweightModelAssistant | None" = None,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._lightweight_assistant = lightweight_assistant

    async def consolidate(self, user_input: str, assistant_reply: str) -> list[str]:
        """提取单轮对话中的候选长期事实。"""

        # 如果有轻量模型可用，使用它来提取事实
        if self._lightweight_assistant and self._lightweight_assistant.available:
            try:
                facts = await self._extract_with_fast_model(user_input, assistant_reply)
            except Exception as exc:
                log.warning("轻量模型提取事实失败，降级到主模型: %s", exc)
                facts = await self._extract_with_main_model(user_input, assistant_reply)
        else:
            facts = await self._extract_with_main_model(user_input, assistant_reply)

        # 如果有轻量模型可用，应用记忆门控过滤
        if self._lightweight_assistant and self._lightweight_assistant.available and facts:
            try:
                facts = await self._lightweight_assistant.memory_gate(
                    user_input, assistant_reply, facts
                )
            except Exception as exc:
                log.warning("记忆门控失败，跳过过滤: %s", exc)

        # 如果提取到事实，应用去重
        if facts:
            existing_facts = self._get_existing_facts()
            if existing_facts and self._lightweight_assistant and self._lightweight_assistant.available:
                try:
                    facts = await self._lightweight_assistant.deduplicate_facts(
                        facts, existing_facts
                    )
                except Exception as exc:
                    log.warning("事实去重失败，跳过: %s", exc)

        # 写入 PENDING.md
        if facts:
            self._memory.append_pending(facts)

        return facts

    async def _extract_with_main_model(
        self, user_input: str, assistant_reply: str
    ) -> list[str]:
        """使用主模型提取事实。"""
        response = await self._provider.chat(
            [
                {"role": "system", "content": CONSOLIDATION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户：{user_input}\n\n"
                        f"助手：{assistant_reply}"
                    ),
                },
            ],
            max_tokens=400,
        )
        return self._parse_facts(response.content)

    async def _extract_with_fast_model(
        self, user_input: str, assistant_reply: str
    ) -> list[str]:
        """使用轻量模型提取事实。"""
        prompt = f"""从以下对话中提取值得长期保留的用户事实。

用户：{user_input}

助手：{assistant_reply}

只提取稳定的偏好、身份背景、长期项目目标、明确承诺或可复用流程。
不要记录临时问题、助手猜测、工具输出、敏感信息或没有事实价值的内容。

只输出 JSON：{{"facts":["事实一","事实二"]}}。没有可保留事实时输出 {{"facts":[]}}。"""

        response = await self._lightweight_assistant._fast_provider.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=400,
        )
        return self._parse_facts(response.content)

    def _get_existing_facts(self) -> list[str]:
        """获取已存在的事实列表（从 PENDING.md 和 MEMORY.md）。"""
        facts = []

        # 读取 PENDING.md
        pending_content = self._memory.read_pending()
        if pending_content:
            for line in pending_content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    facts.append(line[2:])

        # 读取 MEMORY.md
        memory_content = self._memory.read_all()
        if memory_content:
            for line in memory_content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    facts.append(line[2:])

        return facts

    @staticmethod
    def _parse_facts(content: str) -> list[str]:
        """解析并限制模型返回的候选事实。"""

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        raw_facts = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(raw_facts, list):
            return []

        facts: list[str] = []
        for fact in raw_facts:
            if not isinstance(fact, str):
                continue
            normalized = " ".join(fact.split()).strip()
            if normalized and normalized not in facts:
                facts.append(normalized[:300])
            if len(facts) == 5:
                break
        return facts
