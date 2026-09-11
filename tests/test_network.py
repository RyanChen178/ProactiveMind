"""HTTP 重试与退避测试。"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from core.network import (
    DEFAULT_RETRY_EXCEPTIONS,
    RetryPolicy,
    compute_backoff,
    retry,
    retry_call,
    should_retry_status,
)


class RetryPolicyTest(unittest.TestCase):
    """RetryPolicy 参数校验。"""

    def test_defaults(self) -> None:
        p = RetryPolicy()
        self.assertEqual(p.max_attempts, 3)
        self.assertEqual(p.base_delay_s, 0.3)
        self.assertIn(429, p.retry_statuses)
        self.assertIn(503, p.retry_statuses)

    def test_invalid_max_attempts(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_invalid_base_delay(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(base_delay_s=-1)

    def test_max_delay_less_than_base(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(base_delay_s=5.0, max_delay_s=1.0)

    def test_jitter_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(jitter_ratio=1.5)


class ComputeBackoffTest(unittest.TestCase):
    """退避时间计算。"""

    def test_exponential_growth(self) -> None:
        p = RetryPolicy(base_delay_s=0.5, max_delay_s=10.0, jitter_ratio=0.0)
        self.assertAlmostEqual(compute_backoff(p, 1), 0.5)
        self.assertAlmostEqual(compute_backoff(p, 2), 1.0)
        self.assertAlmostEqual(compute_backoff(p, 3), 2.0)
        self.assertAlmostEqual(compute_backoff(p, 4), 4.0)

    def test_clamped_to_max(self) -> None:
        p = RetryPolicy(base_delay_s=1.0, max_delay_s=3.0, jitter_ratio=0.0)
        self.assertAlmostEqual(compute_backoff(p, 1), 1.0)
        self.assertAlmostEqual(compute_backoff(p, 5), 3.0)
        self.assertAlmostEqual(compute_backoff(p, 100), 3.0)

    def test_jitter_within_ratio(self) -> None:
        p = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter_ratio=0.5)
        # 多次采样应在 ±50% 范围内
        for _ in range(20):
            v = compute_backoff(p, 1)
            self.assertGreaterEqual(v, 0.5)
            self.assertLessEqual(v, 1.5)


class ShouldRetryStatusTest(unittest.TestCase):
    """状态码重试判断。"""

    def test_429_is_retryable(self) -> None:
        p = RetryPolicy(max_attempts=3)
        self.assertTrue(should_retry_status(429, p, 1))

    def test_200_is_not_retryable(self) -> None:
        p = RetryPolicy()
        self.assertFalse(should_retry_status(200, p, 1))

    def test_last_attempt_no_retry(self) -> None:
        p = RetryPolicy(max_attempts=3)
        self.assertFalse(should_retry_status(500, p, 3))


class RetryCallTest(unittest.IsolatedAsyncioTestCase):
    """retry_call 主流程。"""

    async def test_success_no_retry(self) -> None:
        sleeps: list[float] = []
        async def fake_sleep(s):
            sleeps.append(s)

        attempts = 0

        async def func():
            nonlocal attempts
            attempts += 1
            return "ok"

        result = await retry_call(
            RetryPolicy(max_attempts=3, base_delay_s=0.1),
            func,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 1)
        self.assertEqual(sleeps, [])

    async def test_retry_on_exception_then_success(self) -> None:
        sleeps: list[float] = []
        async def fake_sleep(s):
            sleeps.append(s)
            await asyncio.sleep(0)  # 让 loop time 推进

        attempts = 0

        async def func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError("simulated")
            return "ok"

        result = await retry_call(
            RetryPolicy(max_attempts=5, base_delay_s=0.05, jitter_ratio=0.0),
            func,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        # 失败 2 次 → 2 次 sleep
        self.assertEqual(len(sleeps), 2)

    async def test_retry_on_status_code(self) -> None:
        """响应 503 应触发重试。"""
        sleeps: list[float] = []

        async def fake_sleep(s):
            sleeps.append(s)

        attempts = 0
        responses: list[int] = [503, 503, 200]

        async def func():
            nonlocal attempts
            status = responses[min(attempts, len(responses) - 1)]
            attempts += 1
            return _FakeResponse(status)

        result = await retry_call(
            RetryPolicy(max_attempts=5, base_delay_s=0.05, jitter_ratio=0.0),
            func,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(attempts, 3)

    async def test_exhausted_attempts_raises(self) -> None:
        async def fake_sleep(_s):
            pass

        attempts = 0

        async def func():
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError(f"fail {attempts}")

        with self.assertRaises(httpx.ConnectError):
            await retry_call(
                RetryPolicy(max_attempts=3, base_delay_s=0.01),
                func,
                sleep_fn=fake_sleep,
            )
        self.assertEqual(attempts, 3)

    async def test_non_retry_exception_propagates(self) -> None:
        async def fake_sleep(_s):
            pass

        async def func():
            raise ValueError("not retryable")

        with self.assertRaises(ValueError):
            await retry_call(
                RetryPolicy(max_attempts=5),
                func,
                sleep_fn=fake_sleep,
            )

    async def test_total_timeout_exits_early(self) -> None:
        """超过 total_timeout_s 应提前停止重试。"""

        attempts = 0

        async def func():
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectError("boom")

        with self.assertRaises(httpx.ConnectError):
            # 使用真实 asyncio.sleep，total_timeout=0.05s 限制下不到 5 次（base=0.05s）
            await retry_call(
                RetryPolicy(
                    max_attempts=100,
                    base_delay_s=0.05,
                    max_delay_s=0.05,
                    jitter_ratio=0.0,
                    total_timeout_s=0.05,
                ),
                func,
            )
        # 应在合理次数内停止（远小于 100）
        self.assertLess(attempts, 10)

    async def test_on_retry_callback_invoked(self) -> None:
        async def fake_sleep(_s):
            pass

        attempts = 0
        retries: list[tuple[int, Any]] = []

        async def func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError("boom")
            return "ok"

        def on_retry(attempt: int, info: Any) -> None:
            retries.append((attempt, info))

        await retry_call(
            RetryPolicy(max_attempts=5, base_delay_s=0.01, jitter_ratio=0.0),
            func,
            sleep_fn=fake_sleep,
            on_retry=on_retry,
        )
        self.assertEqual(len(retries), 2)
        self.assertEqual(retries[0][0], 1)
        self.assertEqual(retries[1][0], 2)


class RetryDecoratorTest(unittest.IsolatedAsyncioTestCase):
    """retry 装饰器。"""

    async def test_decorated_function_retries(self) -> None:
        sleeps: list[float] = []

        async def fake_sleep(s):
            sleeps.append(s)

        attempts = 0

        @retry(RetryPolicy(max_attempts=3, base_delay_s=0.01, jitter_ratio=0.0), sleep_fn=fake_sleep)
        async def fetch():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ConnectError("boom")
            return 42

        result = await fetch()
        self.assertEqual(result, 42)
        self.assertEqual(attempts, 2)


class _FakeResponse:
    """只带 status_code 的伪响应，模拟 httpx.Response。"""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    async def aread(self) -> bytes:
        return b""

    async def read(self) -> bytes:
        return b""


if __name__ == "__main__":
    unittest.main()