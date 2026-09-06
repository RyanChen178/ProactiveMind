"""Telegram Bot 网关测试。"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from gateways.telegram_bot import (
    DEFAULT_POLL_TIMEOUT,
    MAX_MESSAGE_LENGTH,
    TelegramBot,
    _is_authorized,
    _resolve_env,
    _split_message,
    load_telegram_config,
)


def _make_http_client() -> MagicMock:
    """构造一个返回固定响应的 httpx 客户端（async）。"""
    client = MagicMock()

    post_mock = AsyncMock()

    async def fake_post(url: str, json: dict[str, Any] | None = None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"ok": True, "result": []})
        return resp

    post_mock.side_effect = fake_post
    client.post = post_mock
    client.aclose = AsyncMock()
    return client


class _FakeAgent:
    """模拟 MindLoop.run_stream 的可调用对象。"""

    def __init__(self, reply: str = "你好", chunks: list[str] | None = None):
        self.reply = reply
        self.chunks = chunks if chunks is not None else [reply]
        self.received_inputs: list[str] = []

    async def run_stream(self, user_input: str):
        self.received_inputs.append(user_input)
        for chunk in self.chunks:
            yield chunk


class SplitMessageTest(unittest.TestCase):
    """消息分片逻辑。"""

    def test_short_text_returns_single_chunk(self) -> None:
        chunks = _split_message("你好")
        self.assertEqual(chunks, ["你好"])

    def test_exact_limit_returns_single_chunk(self) -> None:
        text = "x" * MAX_MESSAGE_LENGTH
        chunks = _split_message(text)
        self.assertEqual(chunks, [text])

    def test_long_text_splits_within_limit(self) -> None:
        text = "a" * (MAX_MESSAGE_LENGTH + 100)
        chunks = _split_message(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), MAX_MESSAGE_LENGTH)
        # 拼接后内容一致
        self.assertEqual("".join(chunks), text)

    def test_splits_prefer_newline(self) -> None:
        # 在 limit 范围内有换行 → 应在换行处断开
        line = "a" * 2000
        text = (line + "\n") * 10  # 总长 20010
        chunks = _split_message(text, limit=4096)
        # 每个 chunk 应以 "a" 序列（不含被吃掉的换行）开头
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4096)
            self.assertTrue(chunk.startswith("a"))

    def test_empty_text(self) -> None:
        self.assertEqual(_split_message(""), [""])


class AuthorizationTest(unittest.TestCase):
    """用户白名单检查。"""

    def test_empty_allow_all(self) -> None:
        self.assertTrue(_is_authorized(123, []))
        self.assertTrue(_is_authorized(999, []))

    def test_allow_specific_users(self) -> None:
        allow = [100, 200]
        self.assertTrue(_is_authorized(100, allow))
        self.assertTrue(_is_authorized(200, allow))
        self.assertFalse(_is_authorized(300, allow))


class EnvResolveTest(unittest.TestCase):
    """环境变量占位符解析。"""

    def test_resolves_env_var(self) -> None:
        with patch.dict("os.environ", {"MY_BOT_TOKEN": "secret123"}):
            self.assertEqual(_resolve_env("${MY_BOT_TOKEN}"), "secret123")

    def test_missing_env_returns_original(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_env("${MISSING_VAR}"), "${MISSING_VAR}")

    def test_non_env_string(self) -> None:
        self.assertEqual(_resolve_env("plain_string"), "plain_string")


class TelegramBotTest(unittest.IsolatedAsyncioTestCase):
    """TelegramBot 类行为测试。"""

    def _make_bot(
        self,
        agent: _FakeAgent | None = None,
        token: str = "test_token",
        allow_from: list[int] | None = None,
        http_client: AsyncMock | None = None,
    ) -> tuple[TelegramBot, _FakeAgent, AsyncMock]:
        agent = agent or _FakeAgent(reply="pong")
        client = http_client or _make_http_client()
        bot = TelegramBot(
            token=token,
            agent=agent,  # type: ignore[arg-type]
            allow_from=allow_from,
            http_client=client,
        )
        return bot, agent, client

    async def test_send_message_short(self) -> None:
        bot, _, http = self._make_bot()
        await bot.send_message(chat_id=123, text="hi")
        self.assertEqual(http.post.call_count, 1)

    async def test_send_message_splits_long_text(self) -> None:
        bot, _, http = self._make_bot()
        text = "x" * (MAX_MESSAGE_LENGTH + 10)
        await bot.send_message(chat_id=123, text=text)
        self.assertEqual(http.post.call_count, 2)

    async def test_broadcast_sends_to_last_chat(self) -> None:
        bot, _, http = self._make_bot()
        bot._last_chat_id = 555
        await bot.broadcast("主动消息")
        self.assertEqual(http.post.call_count, 1)

    async def test_broadcast_skips_without_last_chat(self) -> None:
        bot, _, http = self._make_bot()
        await bot.broadcast("主动消息")
        self.assertEqual(http.post.call_count, 0)

    async def test_handle_update_calls_agent(self) -> None:
        agent = _FakeAgent(reply="pong")
        bot, _, _ = self._make_bot(agent=agent)

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 100},
                "from": {"id": 999},
                "text": "ping",
            },
        }
        await bot._handle_update(update)

        self.assertEqual(agent.received_inputs, ["ping"])
        self.assertEqual(bot.last_chat_id, 100)

    async def test_handle_update_rejects_unauthorized(self) -> None:
        agent = _FakeAgent(reply="pong")
        bot, _, http = self._make_bot(agent=agent, allow_from=[100])

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 999},
                "from": {"id": 999},
                "text": "ping",
            },
        }
        await bot._handle_update(update)

        # 未授权 → 不调用 agent，不发送消息
        self.assertEqual(agent.received_inputs, [])
        self.assertEqual(http.post.call_count, 0)

    async def test_handle_update_ignores_non_text(self) -> None:
        agent = _FakeAgent(reply="pong")
        bot, _, http = self._make_bot(agent=agent)

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 100},
                "from": {"id": 999},
                "text": "",  # 空文本应忽略
            },
        }
        await bot._handle_update(update)

        self.assertEqual(agent.received_inputs, [])
        self.assertEqual(http.post.call_count, 0)

    async def test_handle_update_splits_long_reply(self) -> None:
        # agent 返回超过限制的回复，应被分片
        agent = _FakeAgent(reply="x" * (MAX_MESSAGE_LENGTH + 50))
        bot, _, http = self._make_bot(agent=agent)

        update = {
            "update_id": 1,
            "message": {
                "chat": {"id": 100},
                "from": {"id": 999},
                "text": "ping",
            },
        }
        await bot._handle_update(update)

        # 应至少调用 2 次 sendMessage
        self.assertGreaterEqual(http.post.call_count, 2)

    async def test_poll_once_processes_updates_and_advances_offset(self) -> None:
        agent = _FakeAgent(reply="pong")
        bot, _, _ = self._make_bot(agent=agent)

        # 模拟 getUpdates 返回 2 条消息
        async def fake_post(url: str, json: dict[str, Any] | None = None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value={
                "ok": True,
                "result": [
                    {
                        "update_id": 10,
                        "message": {
                            "chat": {"id": 1},
                            "from": {"id": 1},
                            "text": "first",
                        },
                    },
                    {
                        "update_id": 11,
                        "message": {
                            "chat": {"id": 1},
                            "from": {"id": 1},
                            "text": "second",
                        },
                    },
                ],
            })
            return resp

        bot._http.post = fake_post

        count = await bot._poll_once()
        self.assertEqual(count, 2)
        self.assertEqual(agent.received_inputs, ["first", "second"])
        self.assertEqual(bot._offset, 12)

    async def test_poll_once_continues_offset_when_get_updates_empty(self) -> None:
        bot, _, _ = self._make_bot()
        bot._offset = 100

        # 没有新 update → offset 不变
        count = await bot._poll_once()
        self.assertEqual(count, 0)
        self.assertEqual(bot._offset, 100)

    async def test_invalid_token_raises(self) -> None:
        with self.assertRaises(ValueError):
            TelegramBot(token="", agent=_FakeAgent())  # type: ignore[arg-type]


class LoadTelegramConfigTest(unittest.TestCase):
    """从 Config 加载 Telegram 配置。"""

    def test_returns_none_when_disabled(self) -> None:
        cfg = MagicMock()
        cfg._raw = {"telegram": {"enabled": False, "token": "x"}}
        self.assertIsNone(load_telegram_config(cfg))

    def test_returns_none_when_missing(self) -> None:
        cfg = MagicMock()
        cfg._raw = {}
        self.assertIsNone(load_telegram_config(cfg))

    def test_loads_enabled_config(self) -> None:
        cfg = MagicMock()
        cfg._raw = {
            "telegram": {
                "enabled": True,
                "token": "abc123",
                "allow_from": ["100", "200", "bad"],
            }
        }
        result = load_telegram_config(cfg)
        self.assertIsNotNone(result)
        self.assertEqual(result["token"], "abc123")
        # "bad" 解析失败被跳过
        self.assertEqual(result["allow_from"], [100, 200])

    def test_resolves_env_token(self) -> None:
        cfg = MagicMock()
        cfg._raw = {
            "telegram": {
                "enabled": True,
                "token": "${TG_TOKEN}",
                "allow_from": [],
            }
        }
        with patch.dict("os.environ", {"TG_TOKEN": "resolved"}):
            result = load_telegram_config(cfg)
            self.assertEqual(result["token"], "resolved")


if __name__ == "__main__":
    unittest.main()