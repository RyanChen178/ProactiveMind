"""web_fetch 工具 —— 抓取 URL 并转为纯文本供 LLM 阅读。

设计：
  - 复用 core.network 的 RetryPolicy / retry_call（指数退避 + 状态码重试）
  - SSRF 防护：仅允许 http/https，拒绝环回与内网地址
  - HTML 转纯文本：标准库 HTMLParser，去 script/style/模板噪音
  - 大小与超时限制，截断时明确标注
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from core.network import RetryPolicy, retry_call

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_BYTES = 200 * 1024  # 200 KB
DEFAULT_MAX_TEXT_CHARS = 8000

_ALLOWED_SCHEMES = ("http", "https")


class FetchBlockedError(ValueError):
    """URL 被 SSRF 防护拒绝。"""


@dataclass
class FetchResult:
    """一次抓取的结果。"""

    url: str
    status: int
    content_type: str
    text: str
    truncated: bool = False

    def to_tool_output(self) -> str:
        lines = [f"URL: {self.url}", f"HTTP {self.status} ({self.content_type})"]
        if self.truncated:
            lines.append(f"[内容超长，已截断至 {len(self.text)} 字符]")
        lines.append("")
        lines.append(self.text)
        return "\n".join(lines)


def _is_blocked_host(host: str) -> bool:
    """拒绝环回 / 内网 / 链路本地地址。"""
    if not host:
        return True
    lowered = host.lower().strip(".")
    if lowered in ("localhost", "localhost.localdomain"):
        return True
    try:
        addr = ipaddress.ip_address(lowered)
    except ValueError:
        addr = None
    if addr is not None:
        return (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        )
    # 域名解析后再核一次（防止域名指向内网）
    try:
        infos = socket.getaddrinfo(lowered, None)
    except OSError:
        return False  # 解析失败交给 httpx 报错
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip.split("%")[0])
        except ValueError:
            continue
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            return True
    return False


def validate_url(url: str) -> str:
    """校验 URL 方案与目标地址，返回规整后的 URL。被拒绝时抛 FetchBlockedError。"""
    if not isinstance(url, str) or not url.strip():
        raise FetchBlockedError("URL 不能为空")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise FetchBlockedError(
            f"仅允许 http/https URL，得到 scheme={parsed.scheme or '(空)'}"
        )
    if not parsed.hostname:
        raise FetchBlockedError("URL 缺少主机名")
    if _is_blocked_host(parsed.hostname):
        raise FetchBlockedError(f"目标地址被禁止访问: {parsed.hostname}")
    return url


class _TextExtractor(HTMLParser):
    """HTML 转纯文本：丢弃 script/style/noscript，块级标签换行，链接附注 href。"""

    _SKIP = {"script", "style", "noscript", "template"}
    _BLOCK = {
        "p", "div", "section", "article", "header", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "br", "hr", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if tag in self._BLOCK:
            self._chunks.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._chunks.append("[")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in self._BLOCK:
            self._chunks.append("\n")
        if tag == "a":
            self._chunks.append("]")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)
            self._chunks.append(" ")

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        # 压缩连续空白但保留换行
        lines = [line.strip() for line in raw.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned


def html_to_text(html: str) -> str:
    """把 HTML 文档转为可读纯文本。"""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        # 解析失败时降级返回去标签文本
        return html
    return extractor.get_text()


class WebFetcher:
    """带重试、限制与 SSRF 防护的 URL 抓取器。"""

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout_s
        self._max_bytes = max_bytes
        self._max_text = max_text_chars
        self._policy = retry_policy or RetryPolicy(max_attempts=3, total_timeout_s=45.0)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "ProactiveMind/0.1 (+web_fetch tool)"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, url: str) -> FetchResult:
        """抓取并转换 URL，失败抛异常（由工具层兜底为错误文本）。"""
        url = validate_url(url)

        async def _do_fetch() -> httpx.Response:
            return await self._client.get(
                url,
                headers={"Accept": "text/html, text/plain, */*;q=0.5"},
            )

        response = await retry_call(self._policy, _do_fetch)
        if response.status_code >= 400:
            return FetchResult(
                url=str(response.url),
                status=response.status_code,
                content_type=response.headers.get("content-type", ""),
                text=f"抓取失败：HTTP {response.status_code}",
            )

        raw = response.content[: self._max_bytes]
        truncated = len(response.content) > self._max_bytes
        content_type = response.headers.get("content-type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()

        body = raw.decode(charset, errors="replace")
        if "html" in content_type.lower():
            text = html_to_text(body)
        else:
            text = body

        if len(text) > self._max_text:
            text = text[: self._max_text]
            truncated = True

        return FetchResult(
            url=str(response.url),
            status=response.status_code,
            content_type=content_type,
            text=text,
            truncated=truncated,
        )


_default_fetcher: WebFetcher | None = None


def _get_default_fetcher() -> WebFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = WebFetcher()
    return _default_fetcher


async def _tool_web_fetch(args: dict[str, Any]) -> str:
    """web_fetch 工具入口。"""
    url = (args.get("url") or "").strip()
    try:
        result = await _get_default_fetcher().fetch(url)
    except FetchBlockedError as exc:
        return f"错误：{exc}"
    except httpx.HTTPError as exc:
        return f"错误：抓取失败 {type(exc).__name__}: {exc}"
    return result.to_tool_output()


def reset_fetcher_for_test() -> None:
    """测试辅助：重置全局 fetcher。"""
    global _default_fetcher
    _default_fetcher = None
