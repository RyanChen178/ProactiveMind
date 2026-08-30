"""上下文压缩——基于模型 context_window 的自动压缩与检查点记录。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mind.context import estimate_message_tokens, build_history_view
from mind.provider import LLMProvider

logger = logging.getLogger(__name__)

SOFT_LIMIT_RATIO = 0.74
SUMMARY_MAX_TOKENS = 8192
KEEP_RECENT_TOKENS = 20000

_SUMMARY_PROMPT = """请为以下对话历史生成结构化摘要。

摘要将替代旧的历史消息，用于保持长对话的上下文连续性。只记录已发生的事实，不要添加猜测。

必须使用以下标题结构：
## 目标
## 约束与偏好
## 进度
### 已完成
### 进行中
### 受阻
## 关键决策
## 下一步
## 关键上下文

保留文件路径、符号名、命令、错误信息、数值和验证结果。省略重复探索、无用日志和协议细节。只输出摘要正文。"""


@dataclass(frozen=True)
class CompactionCheckpoint:
    """压缩检查点——记录一次上下文压缩的元数据。"""

    generation: int
    summary: str
    context_window: int
    soft_limit_tokens: int
    keep_recent_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    compressed_message_count: int
    retained_message_count: int
    timestamp: str
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "summary": self.summary,
            "context_window": self.context_window,
            "soft_limit_tokens": self.soft_limit_tokens,
            "keep_recent_tokens": self.keep_recent_tokens,
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "compressed_message_count": self.compressed_message_count,
            "retained_message_count": self.retained_message_count,
            "timestamp": self.timestamp,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompactionCheckpoint":
        return cls(
            generation=data["generation"],
            summary=data["summary"],
            context_window=data["context_window"],
            soft_limit_tokens=data["soft_limit_tokens"],
            keep_recent_tokens=data["keep_recent_tokens"],
            estimated_tokens_before=data["estimated_tokens_before"],
            estimated_tokens_after=data["estimated_tokens_after"],
            compressed_message_count=data["compressed_message_count"],
            retained_message_count=data["retained_message_count"],
            timestamp=data["timestamp"],
            digest=data["digest"],
        )


class ContextCompactor:
    """上下文压缩器——根据模型 context_window 自动压缩历史消息。"""

    def __init__(
        self,
        provider: LLMProvider,
        context_window: int,
        keep_recent_tokens: int = KEEP_RECENT_TOKENS,
    ) -> None:
        self._provider = provider
        self._context_window = context_window
        self._soft_limit = int(context_window * SOFT_LIMIT_RATIO)
        self._keep_recent_tokens = keep_recent_tokens
        self._generation = 0
        self._active_checkpoint: CompactionCheckpoint | None = None

    @property
    def soft_limit(self) -> int:
        return self._soft_limit

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_checkpoint(self) -> CompactionCheckpoint | None:
        return self._active_checkpoint

    def should_compact(self, messages: list[dict]) -> bool:
        """判断是否需要压缩上下文。"""
        total_tokens = sum(estimate_message_tokens(m) for m in messages)
        return total_tokens >= self._soft_limit

    async def compact(
        self, messages: list[dict], parent_generation: int = 0
    ) -> tuple[list[dict], CompactionCheckpoint]:
        """压缩上下文，返回压缩后的消息列表和检查点。"""
        total_tokens_before = sum(estimate_message_tokens(m) for m in messages)
        retained_messages, compressed_messages = self._split_messages(messages)

        if not compressed_messages:
            logger.warning("没有需要压缩的消息，跳过压缩")
            return messages, self._active_checkpoint or self._create_empty_checkpoint()

        summary = await self._generate_summary(compressed_messages)
        self._generation += 1
        total_tokens_after = sum(estimate_message_tokens(m) for m in retained_messages)
        total_tokens_after += self._estimate_summary_tokens(summary)

        digest = self._compute_digest(compressed_messages)
        checkpoint = CompactionCheckpoint(
            generation=self._generation,
            summary=summary,
            context_window=self._context_window,
            soft_limit_tokens=self._soft_limit,
            keep_recent_tokens=self._keep_recent_tokens,
            estimated_tokens_before=total_tokens_before,
            estimated_tokens_after=total_tokens_after,
            compressed_message_count=len(compressed_messages),
            retained_message_count=len(retained_messages),
            timestamp=datetime.now(timezone.utc).isoformat(),
            digest=digest,
        )
        self._active_checkpoint = checkpoint

        summary_message = {
            "role": "system",
            "content": f"[历史摘要 - 第 {self._generation} 代]\n{summary}",
        }
        compacted_messages = [summary_message] + retained_messages

        logger.info(
            "上下文压缩完成: %d → %d tokens, 压缩 %d 条消息, 保留 %d 条",
            total_tokens_before,
            total_tokens_after,
            len(compressed_messages),
            len(retained_messages),
        )

        return compacted_messages, checkpoint

    def _split_messages(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """分割消息为保留部分和压缩部分。"""
        retained: list[dict] = []
        retained_tokens = 0

        for message in reversed(messages):
            msg_tokens = estimate_message_tokens(message)
            if retained_tokens + msg_tokens > self._keep_recent_tokens:
                break
            retained.append(message)
            retained_tokens += msg_tokens

        retained.reverse()
        compressed = messages[: len(messages) - len(retained)]
        return retained, compressed

    async def _generate_summary(self, messages: list[dict]) -> str:
        """使用 LLM 生成历史消息的结构化摘要。"""
        if not messages:
            return "（无历史消息）"

        history_text = self._format_messages_for_summary(messages)
        prompt_messages = [
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": f"请为以下对话历史生成摘要：\n\n{history_text}"},
        ]

        try:
            response = await self._provider.chat(
                prompt_messages,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            return response.content.strip()
        except Exception as e:
            logger.error("生成摘要失败: %s", e)
            return f"（压缩了 {len(messages)} 条历史消息）"

    def _format_messages_for_summary(self, messages: list[dict]) -> str:
        """将消息列表格式化为可读的文本。"""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"用户: {content}")
            elif role == "assistant":
                parts.append(f"助手: {content}")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                parts.append(f"工具结果 [{tool_call_id}]: {content[:500]}")
        return "\n".join(parts)

    def _estimate_summary_tokens(self, summary: str) -> int:
        """估算摘要占用的 token 数。"""
        return 4 + len(summary) // 4

    def _compute_digest(self, messages: list[dict]) -> str:
        """计算被压缩消息的摘要哈希。"""
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _create_empty_checkpoint(self) -> CompactionCheckpoint:
        """创建一个空的检查点。"""
        return CompactionCheckpoint(
            generation=0,
            summary="",
            context_window=self._context_window,
            soft_limit_tokens=self._soft_limit,
            keep_recent_tokens=self._keep_recent_tokens,
            estimated_tokens_before=0,
            estimated_tokens_after=0,
            compressed_message_count=0,
            retained_message_count=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            digest="",
        )
