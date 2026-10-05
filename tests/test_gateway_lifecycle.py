"""网关资源生命周期测试：幂等关闭 + 所有权 + 关闭后行为。

设计思想：网关资源必须在其所属事件循环内关闭，且
  - aclose 幂等（重复调用不抛错）
  - 只关闭自己创建的资源（注入的 client 归调用方所有）
  - 关闭后 API 调用返回明确失败而非让异常穿透
  - 关闭先停轮询循环，再释放连接
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx

from gateways.telegram_bot import TelegramBot


def _make_http_client() -> MagicMock:
    client = MagicMock()
    post_mock = AsyncMock()

    async def fake_post(url: str, json: dict | None = None):
        # 制造调度点，避免 mock 立即完成导致 run() 紧循环独占事件循环
        await asyncio.sleep(0.001)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"ok": True, "result": []})
        return resp

    post_mock.side_effect = fake_post
    client.post = post_mock
    client.aclose = AsyncMock()
    return client


def _make_agent(reply: str = "ok"):
    agent = MagicMock()

    async def _run_stream(text):
        yield reply

    agent.run_stream = _run_stream
    return agent


class AcloseIdempotentTest(unittest.IsolatedAsyncioTestCase):
    """aclose 重复调用不抛错。"""

    async def test_double_aclose_safe(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        await bot.aclose()
        # 第二次 aclose 应直接返回，不触碰已关闭的 client
        await bot.aclose()
        self.assertTrue(bot._closed)

    async def test_aclose_sets_running_false(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        bot._running = True
        await bot.aclose()
        self.assertFalse(bot._running)


class HttpClientOwnershipTest(unittest.IsolatedAsyncioTestCase):
    """注入的 client 不被 aclose 关闭；自建的会被关闭。"""

    async def test_injected_client_not_closed(self) -> None:
        client = _make_http_client()
        bot = TelegramBot(token="t", agent=_make_agent(), http_client=client)
        await bot.aclose()
        client.aclose.assert_not_called()

    async def test_owned_client_closed(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        await bot.aclose()
        self.assertTrue(bot._http.is_closed)


class ClosedGatewayBehaviorTest(unittest.IsolatedAsyncioTestCase):
    """关闭后 API 调用返回明确失败。"""

    async def test_call_after_close_returns_failure(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        await bot.aclose()
        result = await bot._call("getMe")
        self.assertFalse(result.get("ok"))
        self.assertIn("closed", result.get("error", ""))

    async def test_send_message_after_close_fails(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        await bot.aclose()
        ok = await bot.send_message(123, "hello")
        self.assertFalse(ok)

    async def test_broadcast_after_close_no_crash(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        await bot.aclose()
        bot._last_chat_id = 1
        # 不应抛异常
        await bot.broadcast("msg")


class PollLoopShutdownTest(unittest.IsolatedAsyncioTestCase):
    """aclose 停止轮询循环。"""

    async def test_run_exits_after_aclose(self) -> None:
        client = _make_http_client()
        bot = TelegramBot(token="t", agent=_make_agent(), http_client=client)

        async def _aclose_soon():
            await asyncio.sleep(0.05)
            await bot.aclose()

        closer = asyncio.create_task(_aclose_soon())
        # run() 应在 aclose 后的一轮 poll 内退出
        await asyncio.wait_for(bot.run(), timeout=3.0)
        await closer


class RealClientLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """真实 httpx client 的端到端生命周期。"""

    async def test_real_client_created_and_closed(self) -> None:
        bot = TelegramBot(token="t", agent=_make_agent())
        self.assertFalse(bot._http.is_closed)
        await bot.aclose()
        self.assertTrue(bot._http.is_closed)


if __name__ == "__main__":
    unittest.main()