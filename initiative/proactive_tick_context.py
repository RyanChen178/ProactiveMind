"""主动链路 Tick 上下文 —— 一轮 tick 的元数据容器。

设计思想（与 a20b45af proactive_v2/context.py 中 AgentTickContext 对齐）：

  - 每轮 tick 一个 AgentTickContext 实例
  - 最小字段：tick_id / now_utc / session_key
  - 额外数据通过 dict-style attributes 持有，方便扩展
  - 用 dataclass + field(default_factory) 保证可变性

ProactiveMind 命名（initiative/proactive_tick_context.py）独立于原仓库。
这里只挑核心元数据落地，完整版（30+ 字段）按需再扩展。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def new_tick_id() -> str:
    """生成一个短随机 tick id。"""
    return uuid.uuid4().hex[:8]


@dataclass
class AgentTickContext:
    """一轮 tick 的最小上下文载体。"""

    tick_id: str = field(default_factory=new_tick_id)
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_key: str = ""
    context_as_fallback_open: bool = False
