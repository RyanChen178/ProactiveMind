"""统一 HTTP 重试 + 退避工具 —— ProactiveMind 网络层基础设施。

设计要点：
  - RetryPolicy 数据类：max_attempts / retry_statuses / base_delay / max_delay / jitter
  - retry_async() 装饰器 + retry_call() 直接调用两种风格
  - 指数退避 + 抖动，避免惊群
  - 支持 retry_on 谓词注入（适配不同异常）
  - 总超时 deadline 控制，避免无限重试

用法：
  from core.network import retry_call, RetryPolicy

  policy = RetryPolicy(max_attempts=3, base_delay_s=0.5)
  result = await retry_call(policy, lambda: httpx_client.get(url))
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

import httpx

log = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.TransportError,
    httpx.RemoteProtocolError,
)


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略。"""

    max_attempts: int = 3
    retry_statuses: frozenset[int] = field(default_factory=lambda: DEFAULT_RETRY_STATUSES)
    base_delay_s: float = 0.3
    max_delay_s: float = 5.0
    jitter_ratio: float = 0.2  # 0..1，0 表示无抖动
    total_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")
        if self.base_delay_s < 0:
            raise ValueError("base_delay_s 必须 >= 0")
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s 必须 >= base_delay_s")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise ValueError("jitter_ratio 必须在 0..1 之间")


def compute_backoff(
    policy: RetryPolicy,
    attempt: int,
    *,
    rng: random.Random | None = None,
) -> float:
    """计算第 attempt 次失败后的退避时间（秒）。

    公式：base_delay * 2^(attempt-1)，然后夹到 max_delay 内，
    再叠加 ± jitter_ratio 的随机抖动。
    """
    rng = rng or random
    raw = policy.base_delay_s * (2 ** max(0, attempt - 1))
    capped = min(raw, policy.max_delay_s)
    if policy.jitter_ratio <= 0:
        return capped
    jitter = capped * policy.jitter_ratio * (rng.random() * 2 - 1)
    return max(0.0, capped + jitter)


def should_retry_status(
    status_code: int,
    policy: RetryPolicy,
    attempt: int,
) -> bool:
    """判断 HTTP 状态码是否值得重试。"""
    if attempt >= policy.max_attempts:
        return False
    return status_code in policy.retry_statuses


async def _sleep_with_deadline(
    seconds: float,
    deadline: float | None,
    loop,
    sleep_fn: Callable[[float], Awaitable[None]],
) -> bool:
    """睡眠 seconds 秒，但不超过 deadline。返回是否完整睡眠。"""
    if deadline is not None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        seconds = min(seconds, remaining)
    if seconds > 0:
        await sleep_fn(seconds)
        # sleep_fn 通常是 asyncio.sleep，会推进 loop.time()；
        # 若调用方注入了 fake sleep，需要调用方自己实现时间推进。
    return True


async def retry_call(
    policy: RetryPolicy,
    func: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException | int], None] | None = None,
) -> T:
    """带重试 + 退避地调用 async 函数。

    Args:
        policy: 重试策略
        func: 无参 async 函数（每次重试会重新调用）
        retry_on: 触发重试的异常类型元组
        sleep_fn: 睡眠函数（默认 asyncio.sleep；测试可注入 fake sleep）
        on_retry: 重试回调（attempt, exc_or_status），用于打日志或指标

    Returns:
        func 的返回值

    Raises:
        最后一次失败抛出的异常
    """
    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + policy.total_timeout_s
        if policy.total_timeout_s is not None
        else None
    )
    last_exc: BaseException | None = None
    last_status: int | None = None

    for attempt in range(1, policy.max_attempts + 1):
        if deadline is not None and loop.time() >= deadline:
            break
        result: T | None = None
        raised_exc: BaseException | None = None
        try:
            result = await func()  # type: ignore[assignment]
        except retry_on as exc:
            raised_exc = exc
        except Exception:
            raise

        if raised_exc is not None:
            last_exc = raised_exc
            if on_retry is not None and attempt < policy.max_attempts:
                on_retry(attempt, raised_exc)
            if attempt >= policy.max_attempts:
                raise raised_exc
            if deadline is not None and loop.time() >= deadline:
                raise raised_exc
        else:
            # 进入这里：func() 没抛异常
            status = getattr(result, "status_code", None)
            if (
                status is not None
                and should_retry_status(int(status), policy, attempt)
            ):
                last_status = int(status)
                last_exc = None
                # 读取 body 以便复用连接
                try:
                    if hasattr(result, "aread"):
                        await result.aread()
                    elif hasattr(result, "read"):
                        await result.read()
                except Exception:
                    pass
                if on_retry is not None:
                    on_retry(attempt, int(status))
            else:
                return result  # type: ignore[return-value]

        # 退避
        delay = compute_backoff(policy, attempt)
        if not await _sleep_with_deadline(delay, deadline, loop, sleep_fn):
            break

    # 用尽重试：优先抛原始异常
    if last_exc is not None:
        raise last_exc
    if last_status is not None:
        raise httpx.HTTPStatusError(
            f"max retries exceeded, last status {last_status}",
            request=None,  # type: ignore[arg-type]
            response=None,  # type: ignore[arg-type]
        )
    raise RuntimeError("retry_call exited without result or exception")


def retry(
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, BaseException | int], None] | None = None,
):
    """装饰器版本：把 async 函数包成带重试的版本。"""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: object, **kwargs: object) -> T:
            return await retry_call(
                policy,
                lambda: func(*args, **kwargs),  # type: ignore[arg-type]
                retry_on=retry_on,
                sleep_fn=sleep_fn,
                on_retry=on_retry,
            )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator