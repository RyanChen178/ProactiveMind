"""主动推送限频器测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from initiative.rate_limiter import (
    RateDecision,
    TokenBucketLimiter,
    WindowRateLimiter,
)


class WindowRateLimiterTest(unittest.TestCase):
    """滑动窗口限频器。"""

    def test_under_limit_allows(self) -> None:
        limiter = WindowRateLimiter(max_count=3, window_seconds=60)
        for _ in range(3):
            self.assertTrue(limiter.allow().allowed)
        self.assertEqual(limiter.current_count, 3)

    def test_over_limit_rejects(self) -> None:
        limiter = WindowRateLimiter(max_count=2, window_seconds=60)
        self.assertTrue(limiter.allow().allowed)
        self.assertTrue(limiter.allow().allowed)
        decision = limiter.allow()
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "window_exceeded")
        self.assertGreater(decision.retry_after_s, 0)

    def test_window_slides_expire_old(self) -> None:
        # 手动控制时钟
        now_values = [100.0]
        limiter = WindowRateLimiter(
            max_count=2,
            window_seconds=10.0,
            now_fn=lambda: now_values[0],
        )
        self.assertTrue(limiter.allow().allowed)
        self.assertTrue(limiter.allow().allowed)
        self.assertFalse(limiter.allow().allowed)

        # 时间前进 11 秒，旧事件应过期
        now_values[0] = 111.0
        self.assertTrue(limiter.allow().allowed)
        self.assertEqual(limiter.current_count, 1)

    def test_remaining_count(self) -> None:
        limiter = WindowRateLimiter(max_count=3, window_seconds=60)
        d1 = limiter.allow()
        self.assertEqual(d1.remaining, 2)
        d2 = limiter.allow()
        self.assertEqual(d2.remaining, 1)
        d3 = limiter.allow()
        self.assertEqual(d3.remaining, 0)

    def test_reset_clears_state(self) -> None:
        limiter = WindowRateLimiter(max_count=1, window_seconds=60)
        limiter.allow()
        limiter.reset()
        self.assertEqual(limiter.current_count, 0)
        self.assertTrue(limiter.allow().allowed)

    def test_persistence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rl.json"
            limiter1 = WindowRateLimiter(
                max_count=5, window_seconds=60, storage_path=path,
            )
            limiter1.allow()
            limiter1.allow()

            # 新实例应能读到历史事件
            limiter2 = WindowRateLimiter(
                max_count=5, window_seconds=60, storage_path=path,
            )
            self.assertEqual(limiter2.current_count, 2)

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            WindowRateLimiter(max_count=0, window_seconds=60)
        with self.assertRaises(ValueError):
            WindowRateLimiter(max_count=3, window_seconds=0)


class TokenBucketLimiterTest(unittest.TestCase):
    """令牌桶限频器。"""

    def test_initial_capacity_allows_burst(self) -> None:
        limiter = TokenBucketLimiter(capacity=3, refill_rate=1.0)
        for _ in range(3):
            self.assertTrue(limiter.allow().allowed)
        self.assertFalse(limiter.allow().allowed)

    def test_refill_after_time(self) -> None:
        now_values = [100.0]
        limiter = TokenBucketLimiter(
            capacity=2, refill_rate=1.0,
            now_fn=lambda: now_values[0],
        )
        # 消耗所有令牌
        limiter.allow()
        limiter.allow()
        self.assertFalse(limiter.allow().allowed)

        # 时间前进 1 秒，应补充 1 个令牌
        now_values[0] = 101.0
        self.assertTrue(limiter.allow().allowed)
        self.assertFalse(limiter.allow().allowed)

    def test_retry_after_calculated(self) -> None:
        now_values = [100.0]
        limiter = TokenBucketLimiter(
            capacity=1, refill_rate=2.0,  # 每秒补充 2 个
            now_fn=lambda: now_values[0],
        )
        limiter.allow()
        decision = limiter.allow()
        self.assertFalse(decision.allowed)
        # 补充 1 个令牌需要 0.5 秒
        self.assertAlmostEqual(decision.retry_after_s, 0.5, places=2)

    def test_cost_greater_than_one(self) -> None:
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1.0)
        # 单次消耗 5 个令牌
        self.assertTrue(limiter.allow(cost=5).allowed)
        self.assertEqual(int(limiter.available_tokens), 5)
        # 再消耗 5 个
        self.assertTrue(limiter.allow(cost=5).allowed)
        self.assertFalse(limiter.allow(cost=1).allowed)

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            TokenBucketLimiter(capacity=0, refill_rate=1.0)
        with self.assertRaises(ValueError):
            TokenBucketLimiter(capacity=1, refill_rate=0)


class RateDecisionTest(unittest.TestCase):
    """RateDecision 数据类。"""

    def test_defaults(self) -> None:
        d = RateDecision(allowed=True)
        self.assertEqual(d.reason, "")
        self.assertEqual(d.retry_after_s, 0.0)
        self.assertEqual(d.remaining, 0)