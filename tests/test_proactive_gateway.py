"""主动链路数据网关测试：三路并行预取 + 正文缓存 + 容错。"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from initiative.proactive_gateway import (
    DEFAULT_CONTENT_LIMIT,
    DEFAULT_MAX_CHARS,
    DataGateway,
    GatewayResult,
)


class GatewayResultTest(unittest.TestCase):
    def test_defaults(self) -> None:
        r = GatewayResult()
        self.assertEqual(r.alerts, [])
        self.assertEqual(r.context, [])
        self.assertEqual(r.content_meta, [])
        self.assertEqual(r.content_store, {})


class NoSourceTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_missing_returns_empty(self) -> None:
        gw = DataGateway()
        result = await gw.run()
        self.assertEqual(result.alerts, [])
        self.assertEqual(result.context, [])
        self.assertEqual(result.content_meta, [])
        self.assertEqual(result.content_store, {})


class AlertContextPassthroughTest(unittest.IsolatedAsyncioTestCase):
    async def test_alerts_transparent(self) -> None:
        async def alerts():
            return [{"id": "a1", "content": "告警"}]

        gw = DataGateway(alert_fn=alerts)
        result = await gw.run()
        self.assertEqual(len(result.alerts), 1)
        self.assertEqual(result.alerts[0]["content"], "告警")

    async def test_context_transparent(self) -> None:
        async def ctx():
            return [{"sleep_prob": 0.8}]

        gw = DataGateway(context_fn=ctx)
        result = await gw.run()
        self.assertEqual(result.context[0]["sleep_prob"], 0.8)

    async def test_alerts_exception_degrades(self) -> None:
        async def broken():
            raise RuntimeError("source down")

        gw = DataGateway(alert_fn=broken)
        result = await gw.run()
        self.assertEqual(result.alerts, [])

    async def test_context_exception_degrades(self) -> None:
        async def broken():
            raise RuntimeError("ctx down")

        gw = DataGateway(context_fn=broken)
        result = await gw.run()
        self.assertEqual(result.context, [])

    async def test_one_source_failure_does_not_affect_others(self) -> None:
        async def alerts():
            return [{"id": "a"}]

        async def broken_ctx():
            raise RuntimeError("down")

        async def feed(limit):
            return [{"id": "c1", "url": "https://e.com/1"}]

        gw = DataGateway(
            alert_fn=alerts, context_fn=broken_ctx, feed_fn=feed,
        )
        result = await gw.run()
        self.assertEqual(len(result.alerts), 1)
        self.assertEqual(result.context, [])
        self.assertEqual(len(result.content_meta), 1)


class ContentPathTest(unittest.IsolatedAsyncioTestCase):
    def _feed(self, events):
        async def feed(limit):
            return events[:limit]

        return feed

    def _fetcher(self, mapping=None, default=""):
        async def fetch(url):
            if mapping is not None:
                return mapping.get(url, default)
            return default

        return fetch

    async def test_meta_fields_and_compound_key(self) -> None:
        events = [
            {
                "ack_server": "feed",
                "id": "x1",
                "title": "标题",
                "source": "rss",
                "url": "https://e.com/1",
                "published_at": "2026-01-01",
            }
        ]
        gw = DataGateway(
            feed_fn=self._feed(events),
            web_fetch_fn=self._fetcher(default="正文"),
        )
        result = await gw.run()
        meta = result.content_meta[0]
        self.assertEqual(meta["id"], "feed:x1")  # 复合键
        self.assertEqual(meta["title"], "标题")
        self.assertEqual(meta["source"], "rss")
        self.assertEqual(meta["url"], "https://e.com/1")
        self.assertEqual(meta["published_at"], "2026-01-01")
        self.assertEqual(result.content_store["feed:x1"], "正文")

    async def test_event_id_fallback(self) -> None:
        events = [{"event_id": "e9", "source": "srv", "url": ""}]
        gw = DataGateway(feed_fn=self._feed(events))
        result = await gw.run()
        self.assertEqual(result.content_meta[0]["id"], "srv:e9")

    async def test_body_fetch_failure_stores_empty(self) -> None:
        events = [{"id": "c1", "source": "s", "url": "https://e.com/x"}]

        async def broken(url):
            raise RuntimeError("fetch failed")

        gw = DataGateway(feed_fn=self._feed(events), web_fetch_fn=broken)
        result = await gw.run()
        self.assertEqual(result.content_store["s:c1"], "")

    async def test_no_url_stores_empty(self) -> None:
        events = [{"id": "c1", "source": "s"}]  # 无 url
        gw = DataGateway(
            feed_fn=self._feed(events),
            web_fetch_fn=self._fetcher(default="不该被调用"),
        )
        result = await gw.run()
        self.assertEqual(result.content_store["s:c1"], "")

    async def test_body_truncated(self) -> None:
        events = [{"id": "c1", "source": "s", "url": "https://e.com/x"}]
        gw = DataGateway(
            feed_fn=self._feed(events),
            web_fetch_fn=self._fetcher(default="y" * 5000),
            max_chars=100,
        )
        result = await gw.run()
        self.assertEqual(len(result.content_store["s:c1"]), 100)

    async def test_content_limit_forwarded(self) -> None:
        received_limits: list[int] = []
        events = [{"id": f"c{i}", "source": "s"} for i in range(10)]

        async def feed(limit):
            received_limits.append(limit)
            return events[:limit]  # feed 侧按 limit 截断

        gw = DataGateway(feed_fn=feed, content_limit=3)
        result = await gw.run()
        self.assertEqual(received_limits, [3])
        self.assertEqual(len(result.content_meta), 3)

    async def test_feed_exception_degrades(self) -> None:
        async def broken(limit):
            raise RuntimeError("feed down")

        gw = DataGateway(feed_fn=broken)
        result = await gw.run()
        self.assertEqual(result.content_meta, [])
        self.assertEqual(result.content_store, {})

    async def test_empty_events_short_circuit(self) -> None:
        fetch_called: list[str] = []

        async def fetch(url):
            fetch_called.append(url)
            return "不该被调用"

        async def feed(limit):
            return []

        gw = DataGateway(feed_fn=feed, web_fetch_fn=fetch)
        result = await gw.run()
        self.assertEqual(result.content_meta, [])
        self.assertEqual(fetch_called, [])

    async def test_mixed_success_and_failure(self) -> None:
        events = [
            {"id": "ok1", "source": "s", "url": "https://e.com/1"},
            {"id": "bad", "source": "s", "url": "https://e.com/404"},
            {"id": "ok2", "source": "s", "url": "https://e.com/2"},
        ]

        async def fetch(url):
            if "404" in url:
                return ""  # 空响应视为失败
            return f"body of {url}"

        gw = DataGateway(feed_fn=self._feed(events), web_fetch_fn=fetch)
        result = await gw.run()
        self.assertEqual(result.content_store["s:ok1"], "body of https://e.com/1")
        self.assertEqual(result.content_store["s:bad"], "")
        self.assertEqual(result.content_store["s:ok2"], "body of https://e.com/2")

    async def test_bodies_fetched_in_parallel(self) -> None:
        """并发抓取验证：多条 URL 的抓取重叠执行。"""
        events = [
            {"id": f"c{i}", "source": "s", "url": f"https://e.com/{i}"}
            for i in range(3)
        ]
        active: list[int] = []
        max_concurrent: list[int] = [0]

        async def fetch(url):
            active.append(1)
            max_concurrent[0] = max(max_concurrent[0], len(active))
            await asyncio.sleep(0.05)
            active.pop()
            return "body"

        gw = DataGateway(feed_fn=self._feed(events), web_fetch_fn=fetch)
        result = await gw.run()
        # 至少有两条重叠执行
        self.assertGreaterEqual(max_concurrent[0], 2)
        self.assertEqual(len(result.content_store), 3)


class ThreeWayParallelTest(unittest.IsolatedAsyncioTestCase):
    async def test_sources_overlap(self) -> None:
        """三路数据源并行拉取（重叠执行而非串行）。"""
        active: list[str] = []
        max_concurrent: list[int] = [0]

        def _slow(name):
            async def fn(*args):
                active.append(name)
                max_concurrent[0] = max(max_concurrent[0], len(active))
                await asyncio.sleep(0.05)
                active.remove(name)
                return [{"from": name}]

            return fn

        gw = DataGateway(
            alert_fn=_slow("a"), context_fn=_slow("c"), feed_fn=_slow("f"),
        )
        result = await gw.run()
        self.assertGreaterEqual(max_concurrent[0], 2)
        self.assertEqual(len(result.alerts), 1)
        self.assertEqual(len(result.context), 1)
        self.assertEqual(len(result.content_meta), 1)


class DefaultsTest(unittest.TestCase):
    def test_module_constants(self) -> None:
        self.assertEqual(DEFAULT_MAX_CHARS, 8000)
        self.assertEqual(DEFAULT_CONTENT_LIMIT, 5)


if __name__ == "__main__":
    unittest.main()
