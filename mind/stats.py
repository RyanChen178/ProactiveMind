"""Turn 统计 —— 记录每轮对话的 token 用量、延迟、工具调用。

TurnStats 累积全部历史 turn；StatsCollector 提供汇总视图。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TurnRecord:
    """一轮对话的统计记录。"""

    turn_id: int
    session_id: str
    user_input: str
    assistant_reply: str
    tool_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TurnStats:
    """Turn 统计收集器。"""

    def __init__(self, max_records: int = 500) -> None:
        self._records: list[TurnRecord] = []
        self._max_records = max_records
        self._turn_counter = 0

    def begin_turn(self, session_id: str, user_input: str) -> int:
        """开始一轮对话，返回 turn_id。"""
        self._turn_counter += 1
        return self._turn_counter

    def record(
        self,
        session_id: str,
        user_input: str,
        assistant_reply: str,
        tool_calls: list[str] | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: float = 0.0,
    ) -> TurnRecord:
        """记录一轮对话的完整统计。"""
        self._turn_counter += 1
        record = TurnRecord(
            turn_id=self._turn_counter,
            session_id=session_id,
            user_input=user_input[:200],
            assistant_reply=assistant_reply[:200],
            tool_calls=tool_calls or [],
            prompt_tokens=(usage or {}).get("prompt_tokens", 0),
            completion_tokens=(usage or {}).get("completion_tokens", 0),
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]
        return record

    def summary(self) -> dict:
        """返回汇总统计。"""
        total = len(self._records)
        if total == 0:
            return {
                "total_turns": 0,
                "total_tokens": 0,
                "total_tool_calls": 0,
                "avg_latency_ms": 0,
            }
        total_tokens = sum(r.total_tokens for r in self._records)
        total_tool_calls = sum(len(r.tool_calls) for r in self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / total
        return {
            "total_turns": total,
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "avg_latency_ms": round(avg_latency, 1),
        }

    def recent(self, n: int = 10) -> list[TurnRecord]:
        """返回最近 n 轮记录。"""
        return self._records[-n:] if n > 0 else []

    @property
    def records(self) -> list[TurnRecord]:
        return list(self._records)
