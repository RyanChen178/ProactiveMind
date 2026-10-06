"""主动链路契约 —— metrics 归一化与时间格式化。

设计思想（与 a20b45af proactive_v2/contracts.py 对齐）：

  - normalize_metrics：把任意 dict 压成可观测上报的小型载荷
    - 限制 key 数量（防止爆炸）
    - 字符串超长截断
    - 非基本类型走 json.dumps 兜底
    - 超出配额时记录 _truncated_keys 计数
  - looks_like_time_key：识别时间相关 key
  - resolve_tz：字符串 → tzinfo 解析（zoneinfo）
  - format_local_time：ISO 时间转本地时区字符串

ProactiveMind 命名（initiative/proactive_contracts.py）独立，
机制与文件归一化策略同构。
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MAX_METRICS_KEYS = 8
MAX_METRICS_VALUE_STR_LEN = 60
_TIME_KEY_SUFFIXES = ("_at", "_time", "_ts")
_TIME_KEYS = frozenset({
    "last_seen", "updated_at", "published_at", "timestamp", "ts",
})


def trim_text(text: str, limit: int) -> str:
    """超过 limit 截断并加 '...' 标记。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def normalize_metrics(metrics: Any) -> dict[str, Any] | None:
    """把任意 dict 归一化为可观测上报负载。

    - 限制最多 MAX_METRICS_KEYS 个 key
    - 字符串超长截断
    - 非基本类型用 json.dumps 序列化
    - 超额时记录 _truncated_keys 计数
    - 空输入返回 None
    """
    if not isinstance(metrics, dict) or not metrics:
        return None

    normalized: dict[str, Any] = {}
    items = list(metrics.items())
    for key, value in items[:MAX_METRICS_KEYS]:
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, str):
            normalized[key_text] = trim_text(value, MAX_METRICS_VALUE_STR_LEN)
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            normalized[key_text] = value
            continue

        text = json.dumps(value, ensure_ascii=False)
        normalized[key_text] = trim_text(text, MAX_METRICS_VALUE_STR_LEN)

    truncated = len(items) - MAX_METRICS_KEYS
    if truncated > 0:
        normalized["_truncated_keys"] = truncated

    return normalized or None


def looks_like_time_key(key: str) -> bool:
    """识别 key 是否像时间字段（白名单或 _at/_time/_ts 后缀）。"""
    return key in _TIME_KEYS or key.endswith(_TIME_KEY_SUFFIXES)


def resolve_tz(value: str | tzinfo | None) -> tzinfo | None:
    """把字符串（zoneinfo 名）或 tzinfo 转 tzinfo 对象。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return ZoneInfo(text)
        except ZoneInfoNotFoundError:
            return None
    return value


def format_local_time(raw: str, local_tz: str | tzinfo | None = None) -> str | None:
    """把 ISO 时间字符串转成本地时区格式化字符串。

    失败时返回 None（如 raw 非合法 ISO 字符串或无 tz 后缀）。
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return None
    tz = resolve_tz(local_tz)
    local_dt = dt.astimezone(tz) if tz is not None else dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %z")
