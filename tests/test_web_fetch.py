"""web_fetch 工具测试。"""

from __future__ import annotations

import socket
import unittest
import unittest.mock
from unittest.mock import MagicMock

import httpx

from mind.tools import build_core_tools
from mind.web_fetch import (
    FetchBlockedError,
    FetchResult,
    WebFetcher,
    _tool_web_fetch,
    html_to_text,
    reset_fetcher_for_test,
    validate_url,
)


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _public_dns(host, *args, **kwargs):
    """伪造公网 DNS 解析，隔离测试环境对本机 DNS 的依赖。"""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


class ValidateUrlTest(unittest.TestCase):
    """URL 安全校验。"""

    def setUp(self) -> None:
        # 域名用例走伪造公网 DNS，避免本机 DNS 劫持导致误判
        patcher = unittest.mock.patch(
            "mind.web_fetch.socket.getaddrinfo", side_effect=_public_dns
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_allows_https(self) -> None:
        self.assertEqual(
            validate_url("https://example.com/page"),
            "https://example.com/page",
        )

    def test_allows_http(self) -> None:
        self.assertEqual(validate_url("http://example.com"), "http://example.com")

    def test_rejects_domain_resolving_to_private_ip(self) -> None:
        def _private_dns(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]

        with unittest.mock.patch(
            "mind.web_fetch.socket.getaddrinfo", side_effect=_private_dns
        ):
            with self.assertRaises(FetchBlockedError):
                validate_url("https://internal.example.com/")

    def test_rejects_empty(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("   ")

    def test_rejects_ftp_scheme(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("ftp://example.com/file")

    def test_rejects_file_scheme(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("file:///etc/passwd")

    def test_rejects_no_scheme(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("example.com/no-scheme")

    def test_rejects_no_host(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("https:///path-only")

    def test_rejects_localhost(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("https://localhost:8080/admin")

    def test_rejects_loopback_ip(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("http://127.0.0.1/")

    def test_rejects_private_ip(self) -> None:
        for ip in ("10.0.0.1", "192.168.1.1", "172.16.0.1"):
            with self.assertRaises(FetchBlockedError):
                validate_url(f"http://{ip}/")

    def test_rejects_link_local(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("http://169.254.169.254/latest/meta-data")

    def test_rejects_ipv6_loopback(self) -> None:
        with self.assertRaises(FetchBlockedError):
            validate_url("http://[::1]/")


class HtmlToTextTest(unittest.TestCase):
    """HTML 转纯文本。"""

    def test_strips_script_and_style(self) -> None:
        html = "<html><body><style>.x{}</style><script>bad()</script><p>正文</p></body></html>"
        text = html_to_text(html)
        self.assertIn("正文", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn(".x{}", text)

    def test_block_tags_break_lines(self) -> None:
        html = "<p>第一段</p><p>第二段</p>"
        text = html_to_text(html)
        lines = text.splitlines()
        self.assertIn("第一段", lines)
        self.assertIn("第二段", lines)

    def test_link_text_kept(self) -> None:
        html = '<p>参见 <a href="/doc">文档</a> 了解详情</p>'
        text = html_to_text(html)
        self.assertIn("文档", text)
        self.assertIn("了解详情", text)

    def test_collapses_whitespace(self) -> None:
        html = "<p>hello    \n   world</p>"
        text = html_to_text(html)
        self.assertIn("hello world", text)

    def test_plain_text_passthrough(self) -> None:
        # 非 HTML 输入也不应崩溃
        self.assertIn("plain", html_to_text("plain text"))

    def test_noscript_skipped(self) -> None:
        html = "<noscript>需要 JS</noscript><p>内容</p>"
        text = html_to_text(html)
        self.assertNotIn("需要 JS", text)
        self.assertIn("内容", text)


def _client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


class WebFetcherTest(unittest.IsolatedAsyncioTestCase):
    """WebFetcher 抓取行为（MockTransport 注入）。"""

    def setUp(self) -> None:
        patcher = unittest.mock.patch(
            "mind.web_fetch.socket.getaddrinfo", side_effect=_public_dns
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        reset_fetcher_for_test()

    async def test_fetch_html_converts_to_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content="<html><body><h1>标题</h1><p>段落内容</p></body></html>".encode(),
            )

        fetcher = WebFetcher(client=_client_with(handler))
        result = await fetcher.fetch("https://example.com/doc")
        self.assertEqual(result.status, 200)
        self.assertIn("标题", result.text)
        self.assertIn("段落内容", result.text)
        self.assertFalse(result.truncated)

    async def test_fetch_plain_text_passthrough(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=utf-8"},
                content="hello raw".encode(),
            )

        fetcher = WebFetcher(client=_client_with(handler))
        result = await fetcher.fetch("https://example.com/a.txt")
        self.assertEqual(result.text, "hello raw")

    async def test_fetch_404_reports_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        fetcher = WebFetcher(client=_client_with(handler))
        result = await fetcher.fetch("https://example.com/missing")
        self.assertEqual(result.status, 404)
        self.assertIn("抓取失败", result.text)

    async def test_fetch_truncates_long_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="x" * 10000)

        fetcher = WebFetcher(client=_client_with(handler), max_text_chars=100)
        result = await fetcher.fetch("https://example.com/big")
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.text), 100)

    async def test_fetch_retries_on_500_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500)
            return httpx.Response(200, text="recovered")

        fetcher = WebFetcher(
            client=_client_with(handler),
            retry_policy=None,
        )
        # 用无抖动短退避策略
        from core.network import RetryPolicy

        fetcher._policy = RetryPolicy(
            max_attempts=3, base_delay_s=0.001, max_delay_s=0.001, jitter_ratio=0.0
        )
        result = await fetcher.fetch("https://example.com/flaky")
        self.assertEqual(result.status, 200)
        self.assertEqual(result.text, "recovered")
        self.assertEqual(calls["n"], 2)

    async def test_fetch_blocked_url_raises(self) -> None:
        fetcher = WebFetcher(client=_client_with(lambda r: httpx.Response(200)))
        with self.assertRaises(FetchBlockedError):
            await fetcher.fetch("http://127.0.0.1:6322/health")


class ToolEntryPointTest(unittest.IsolatedAsyncioTestCase):
    """_tool_web_fetch 工具入口。"""

    def setUp(self) -> None:
        patcher = unittest.mock.patch(
            "mind.web_fetch.socket.getaddrinfo", side_effect=_public_dns
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        reset_fetcher_for_test()

    async def test_missing_url_argument(self) -> None:
        reset_fetcher_for_test()
        result = await _tool_web_fetch({})
        self.assertIn("错误", result)

    async def test_blocked_url_returns_error_text(self) -> None:
        result = await _tool_web_fetch({"url": "http://localhost/secret"})
        self.assertIn("错误", result)
        self.assertIn("禁止", result)

    async def test_success_output_includes_url_and_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="page body")

        # 注入 mock client 的 fetcher
        import mind.web_fetch as wf

        wf._default_fetcher = WebFetcher(client=_client_with(handler))
        result = await _tool_web_fetch({"url": "https://example.com/x"})
        self.assertIn("https://example.com/x", result)
        self.assertIn("HTTP 200", result)
        self.assertIn("page body", result)


class FetchResultTest(unittest.TestCase):
    """FetchResult 输出格式。"""

    def test_to_tool_output_with_truncation_note(self) -> None:
        r = FetchResult(
            url="https://e.com/a",
            status=200,
            content_type="text/html",
            text="body",
            truncated=True,
        )
        out = r.to_tool_output()
        self.assertIn("URL: https://e.com/a", out)
        self.assertIn("HTTP 200", out)
        self.assertIn("已截断", out)


class BuildCoreToolsTest(unittest.TestCase):
    """web_fetch 注册进内置工具集。"""

    def test_web_fetch_registered(self) -> None:
        registry = build_core_tools(MagicMock())
        names = {s["function"]["name"] for s in registry.get_schemas()}
        self.assertIn("web_fetch", names)
        self.assertIn("shell", names)
        self.assertIn("memorize", names)
        self.assertIn("recall", names)


if __name__ == "__main__":
    unittest.main()