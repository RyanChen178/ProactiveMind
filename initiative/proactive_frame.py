"""主动链路 Frame —— 一轮 tick 的数据载体。

设计思想（与 a20b45af proactive_v2/frame.py 对齐）：

  - 一轮主动决策 = 一个 ProactiveFrame
  - 三段式结构：
      input  —— 本轮只读输入（session_key / 启动时间）
      slots  —— 模块间共享状态字典（gate 判定、prompt 注入、metrics 等）
      output —— 一轮 tick 的可写最终结果
  - slots 在管线模块之间像 scratch memory 一样共享，
    各模块按约定 key 读 / 写
  - frozen=True 保证 input 在 tick 内不被意外改写
  - new_frame() 工厂封装 started_at 取值

ProactiveMind 命名与文件名路径独立（路径 /proactive_frame.py），
路径与 dataclass 一致，机制同构。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _empty_slots() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ProactiveTickInput:
    """一轮 tick 的只读输入。"""

    session_key: str
    started_at: datetime


@dataclass
class ProactiveTickResult:
    """一轮 tick 的可写输出。"""

    base_score: float | None = None


@dataclass
class ProactiveFrame:
    """主动链路的核心数据载体——三段结构 input/slots/output。"""

    input: ProactiveTickInput
    slots: dict[str, Any] = field(default_factory=_empty_slots)
    output: ProactiveTickResult | None = None


def new_frame(
    session_key: str,
    slots: Mapping[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> ProactiveFrame:
    """构造一个新 Frame（started_at 默认取当前 UTC 时间）。"""
    return ProactiveFrame(
        input=ProactiveTickInput(
            session_key=session_key,
            started_at=now or datetime.now(timezone.utc),
        ),
        slots=dict(slots or {}),
    )


# 命名兼容旧调用方
new_proactive_frame = new_frame