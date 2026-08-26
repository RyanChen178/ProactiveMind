"""对话候选记忆提取与待归档缓冲。"""

from __future__ import annotations

import json

from mind.memory import MemoryStore
from mind.provider import LLMProvider

CONSOLIDATION_PROMPT = """\
你负责从一轮用户与助手对话中提取值得长期保留的用户事实。
只保留稳定的偏好、身份背景、长期项目目标、明确承诺或可复用流程；
不要记录临时问题、助手猜测、工具输出、敏感信息或没有事实价值的内容。
只输出 JSON：{"facts":["事实一","事实二"]}。没有可保留事实时输出 {"facts":[]}。"""


class MemoryConsolidator:
    """将模型提取的候选事实写入 PENDING.md。"""

    def __init__(self, provider: LLMProvider, memory: MemoryStore) -> None:
        self._provider = provider
        self._memory = memory

    async def consolidate(self, user_input: str, assistant_reply: str) -> list[str]:
        """提取单轮对话中的候选长期事实。"""

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
        facts = self._parse_facts(response.content)
        self._memory.append_pending(facts)
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
