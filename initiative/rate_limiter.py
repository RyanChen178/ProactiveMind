"""主动推送限频器 —— 避免短时间内重复打扰用户。

两种策略：
  - WindowRateLimiter：滑动时间窗口计数（默认 60 秒最多 3 条推送）
  - TokenBucketLimiter：令牌桶（限制平均速率 + 突发容量）

设计要点：
  - 内存状态 + 原子更新
  - 持久化到 workspace / rate_limit.json（跨重启保留计数）
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)


@dataclass
class RateDecision:
    """限频决策结果。"""

    allowed: bool
    reason: str = ""
    retry_after_s: float = 0.0
    remaining: int = 0


class WindowRateLimiter:
    """滑动窗口限频器。

    在 window_seconds 窗口内最多允许 max_count 次推送。
    每次允许/拒绝后更新窗口中的事件列表；窗口外的旧事件被清理。
    """

    def __init__(
        self,
        max_count: int = 3,
        window_seconds: float = 60.0,
        *,
        now_fn=time.time,
        storage_path: Path | None = None,
    ) -> None:
        if max_count <= 0:
            raise ValueError("max_count 必须 > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds 必须 > 0")
        self._max_count = max_count
        self._window = window_seconds
        self._now = now_fn
        self._events: deque[float] = deque()
        self._lock = Lock()
        self._storage_path = storage_path
        if storage_path is not None:
            self._load()

    def allow(self) -> RateDecision:
        """检查并记录一次推送事件。

        Returns:
            RateDecision：allowed=True 时事件已计入；allowed=False 表示触发限频。
        """
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            if len(self._events) >= self._max_count:
                # 计算最早事件何时移出窗口
                oldest = self._events[0]
                retry_after = max(0.0, oldest + self._window - now)
                self._persist()
                return RateDecision(
                    allowed=False,
                    reason="window_exceeded",
                    retry_after_s=retry_after,
                    remaining=0,
                )
            self._events.append(now)
            self._persist()
            return RateDecision(
                allowed=True,
                remaining=self._max_count - len(self._events),
            )

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._persist()

    @property
    def current_count(self) -> int:
        with self._lock:
            self._evict_expired(self._now())
            return len(self._events)

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "events": list(self._events),
                "saved_at": self._now(),
            }
            self._storage_path.write_text(
                json.dumps(payload), encoding="utf-8",
            )
        except OSError as exc:
            log.warning("限频状态持久化失败: %s", exc)

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            events = data.get("events") or []
            if isinstance(events, list):
                # 仅保留 window 内的
                cutoff = self._now() - self._window
                self._events = deque(t for t in events if isinstance(t, (int, float)) and t >= cutoff)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("限频状态加载失败: %s", exc)


class TokenBucketLimiter:
    """令牌桶限频器。

    - capacity：桶容量（突发上限）
    - refill_rate：每秒补充的令牌数
    """

    def __init__(
        self,
        capacity: int = 5,
        refill_rate: float = 1.0,
        *,
        now_fn=time.time,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须 > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate 必须 > 0")
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._now = now_fn
        self._tokens = float(capacity)
        self._last_refill = now_fn()
        self._lock = Lock()

    def allow(self, cost: float = 1.0) -> RateDecision:
        with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return RateDecision(
                    allowed=True,
                    remaining=int(self._tokens),
                )
            # 计算补充到 cost 所需时间
            need = cost - self._tokens
            retry_after = need / self._refill_rate
            return RateDecision(
                allowed=False,
                reason="token_exhausted",
                retry_after_s=retry_after,
                remaining=int(self._tokens),
            )

    def _refill(self) -> None:
        now = self._now()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(
            float(self._capacity),
            self._tokens + elapsed * self._refill_rate,
        )
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens