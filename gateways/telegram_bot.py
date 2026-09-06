"""Telegram Bot 渠道 —— 通过 Bot API 与 Telegram 用户对话。

工作方式：
  - 启动时通过 getUpdates 长轮询接收用户消息
  - 用户发消息 → 调用 MindLoop 流式输出 → 累积为完整回复后 sendMessage
  - 支持主动推送：broadcast(text) 把消息发送给最近一次聊天的用户

配置（config.toml）：
  [telegram]
  enabled = true
  token = "${TELEGRAM_BOT_TOKEN}"
  allow_from = ["123456789"]   # 允许的用户 ID 列表；空表示允许所有人

依赖：httpx（已通过 requirements.txt 安装）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

from mind.loop import MindLoop

log = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE_LENGTH = 4096
DEFAULT_POLL_TIMEOUT = 30
DEFAULT_RETRY_DELAY = 5.0


def _is_authorized(user_id: int, allow_from: list[int]) -> bool:
    """检查用户 ID 是否在白名单中。空列表表示允许所有。"""
    if not allow_from:
        return True
    return user_id in allow_from


def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """将长文本按 limit 切分，优先在换行/空格处断开。"""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # 在 limit 范围内寻找最近的换行或空格
        cut_at = remaining.rfind("\n", 0, limit)
        if cut_at < limit // 2:
            cut_at = remaining.rfind(" ", 0, limit)
        if cut_at < limit // 2:
            cut_at = limit
        chunks.append(remaining[:cut_at])
        remaining = remaining[cut_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class TelegramBot:
    """Telegram Bot 长轮询客户端。"""

    def __init__(
        self,
        token: str,
        agent: MindLoop,
        *,
        allow_from: list[int] | None = None,
        poll_timeout: int = DEFAULT_POLL_TIMEOUT,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not token:
            raise ValueError("Telegram bot token 不能为空")

        self._token = token
        self._agent = agent
        self._allow_from: list[int] = list(allow_from or [])
        self._poll_timeout = poll_timeout
        self._retry_delay = retry_delay
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self._owns_http = http_client is None

        self._running = False
        self._offset: int | None = None
        self._last_chat_id: int | None = None  # 主动推送目标

    async def aclose(self) -> None:
        """关闭 HTTP 客户端。"""
        self._running = False
        if self._owns_http:
            await self._http.aclose()

    async def _call(self, method: str, **params: Any) -> dict[str, Any]:
        """调用 Telegram Bot API 并返回 result 字段。"""
        url = TELEGRAM_API_BASE.format(token=self._token, method=method)
        params = {k: v for k, v in params.items() if v is not None}
        try:
            resp = await self._http.post(url, json=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            log.warning("Telegram API 调用失败: %s", exc)
            return {"ok": False, "error": str(exc)}
        if not payload.get("ok"):
            log.warning("Telegram API 返回错误: %s", payload.get("description"))
        return payload

    async def send_message(self, chat_id: int, text: str) -> bool:
        """向指定 chat 发送消息；长文本自动分片。"""
        if not text:
            return True
        for chunk in _split_message(text):
            payload = await self._call(
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                parse_mode=None,
            )
            if not payload.get("ok"):
                return False
        return True

    async def broadcast(self, text: str) -> None:
        """主动推送：发送给最近一次聊天的用户。"""
        if self._last_chat_id is None:
            log.debug("Telegram 主动推送跳过：无最近聊天用户")
            return
        ok = await self.send_message(self._last_chat_id, text)
        if ok:
            log.info("Telegram 主动推送成功 chat_id=%s", self._last_chat_id)

    @property
    def last_chat_id(self) -> int | None:
        return self._last_chat_id

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """处理单条 Telegram update。"""
        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        user = message.get("from") or {}
        user_id = user.get("id")
        text = message.get("text") or ""

        if chat_id is None or user_id is None:
            return
        if not text:
            return
        if not _is_authorized(int(user_id), self._allow_from):
            log.info("Telegram 拒绝未授权用户: id=%s", user_id)
            return

        self._last_chat_id = int(chat_id)
        log.info("Telegram 收到消息 chat_id=%s len=%d", chat_id, len(text))

        # 调用 Agent 流式输出
        chunks: list[str] = []
        try:
            async for piece in self._agent.run_stream(text):
                chunks.append(piece)
        except Exception as exc:
            log.exception("Agent 处理失败: %s", exc)
            await self.send_message(chat_id, f"[错误] {exc}")
            return

        reply = "".join(chunks).strip()
        if not reply:
            reply = "（无回复）"
        await self.send_message(chat_id, reply)

    async def _poll_once(self) -> int:
        """执行一轮长轮询；返回本轮处理的 update 数量。"""
        params: dict[str, Any] = {"timeout": self._poll_timeout}
        if self._offset is not None:
            params["offset"] = self._offset

        payload = await self._call("getUpdates", **params)
        result = payload.get("result") or []
        for update in result:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                # offset 单调递增，确保下次拉取不重复
                if self._offset is None or update_id >= self._offset:
                    self._offset = update_id + 1
            await self._handle_update(update)
        return len(result)

    async def run(self) -> None:
        """启动长轮询主循环。"""
        self._running = True
        log.info(
            "Telegram Bot 启动 allow_from=%s offset=%s",
            self._allow_from or "*", self._offset,
        )
        # 通过 getMe 验证 token 有效性
        me = await self._call("getMe")
        if not me.get("ok"):
            log.error("Telegram getMe 失败，请检查 token")
            self._running = False
            return
        bot_info = me.get("result") or {}
        log.info(
            "Telegram Bot 已就绪 username=%s id=%s",
            bot_info.get("username"), bot_info.get("id"),
        )

        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Telegram 轮询异常: %s；%.1fs 后重试", exc, self._retry_delay)
                await asyncio.sleep(self._retry_delay)
        log.info("Telegram Bot 停止")


def _resolve_env(value: str) -> str:
    """解析 ${ENV_VAR} 形式的占位符。"""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        env_key = value[2:-1].strip()
        return os.environ.get(env_key, value)
    return value


def load_telegram_config(config: Any) -> dict[str, Any] | None:
    """从 Config 对象读取 telegram 配置段。

    Returns:
        解析后的 dict；未启用时返回 None。
    """
    raw = getattr(config, "_raw", None) or {}
    telegram_cfg = raw.get("telegram") if isinstance(raw, dict) else None
    if not telegram_cfg:
        return None
    enabled = bool(telegram_cfg.get("enabled", False))
    if not enabled:
        return None

    token = _resolve_env(telegram_cfg.get("token", ""))
    allow_raw = telegram_cfg.get("allow_from") or []
    allow_from: list[int] = []
    if isinstance(allow_raw, list):
        for item in allow_raw:
            try:
                allow_from.append(int(item))
            except (TypeError, ValueError):
                continue

    return {
        "token": token,
        "allow_from": allow_from,
    }


async def run_telegram_gateway(agent: MindLoop, config: Any) -> None:
    """从 config 加载 Telegram 配置并启动 Bot。"""
    tg_cfg = load_telegram_config(config)
    if tg_cfg is None:
        log.warning("Telegram 渠道未启用")
        return
    bot = TelegramBot(
        token=tg_cfg["token"],
        agent=agent,
        allow_from=tg_cfg["allow_from"],
    )
    try:
        await bot.run()
    finally:
        await bot.aclose()