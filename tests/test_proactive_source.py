"""主动链路数据源模块测试。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from initiative.data_sources import DataSourceManager
from initiative.proactive_frame import new_frame
from initiative.proactive_source import (
    SLOT_ALERTS,
    SLOT_CONTENT_META,
    SLOT_CONTEXT,
    SLOT_FALLBACK_OPEN,
    ProactiveSource,
    ProactiveSourceModule,
    _split_compound_key,
)


class SplitCompoundKeyTest(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(
            _split_compound_key("feed:abc123"), ("feed", "abc123")
        )

    def test_no_colon(self) -> None:
        self.assertIsNone(_split_compound_key("nocolon"))

    def test_empty_parts(self) -> None:
        self.assertIsNone(_split_compound_key(":id"))
        self.assertIsNone(_split_compound_key("src:"))
        self.assertIsNone(_split_compound_key(""))


class ProactiveSourceTest(unittest.IsolatedAsyncioTestCase):
    def _manager(self):
        m = DataSourceManager()
        return m

    async def test_alerts_passthrough(self) -> None:
        m = self._manager()
        m.alert_source.add_alert("系统告警")
        src = ProactiveSource(m)
        alerts = await src.fetch_alerts()
        # AlertDataSource.fetch 返回 DataSourceItem
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["content"], "系统告警")
        self.assertEqual(alerts[0]["priority"], 1.0)

    async def test_alerts_source_missing(self) -> None:
        src = ProactiveSource(MagicMock(spec=[]))
        self.assertEqual(await src.fetch_alerts(), [])

    async def test_alerts_exception_degrades(self) -> None:
        ds = MagicMock()
        broken = MagicMock()
        broken.fetch = AsyncMock(side_effect=RuntimeError("down"))
        ds.alert_source = broken
        src = ProactiveSource(ds)
        self.assertEqual(await src.fetch_alerts(), [])

    async def test_content_limit(self) -> None:
        m = self._manager()
        m.content_source._score_threshold = 0.0  # 未评分内容放行
        for i in range(10):
            m.content_source.add_content(f"条目 {i}")
        src = ProactiveSource(m, content_limit=3)
        items = await src.fetch_contents()
        self.assertEqual(len(items), 3)

    async def test_content_limit_zero(self) -> None:
        m = self._manager()
        m.content_source.add_content("唯一")
        src = ProactiveSource(m, content_limit=0)
        self.assertEqual(await src.fetch_contents(), [])

    async def test_context_gated_by_fallback(self) -> None:
        m = self._manager()
        # priority=1.0 保证 context 路必出
        m.context_source.add_context("背景", priority=1.0)
        src = ProactiveSource(m)
        # 未开兜底 → 空
        self.assertEqual(await src.fetch_context(fallback_open=False), [])
        # 开兜底 → 有数据
        items = await src.fetch_context(fallback_open=True)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "背景")


class ProactiveSourceModuleTest(unittest.IsolatedAsyncioTestCase):
    def _module(self, manager=None, **kwargs):
        manager = manager or DataSourceManager()
        return ProactiveSourceModule(
            ProactiveSource(manager, **kwargs)
        )

    async def test_writes_three_slots(self) -> None:
        m = DataSourceManager()
        m.content_source._score_threshold = 0.0
        m.alert_source.add_alert("告警")
        m.content_source.add_content("内容")
        m.context_source.add_context("背景", priority=1.0)
        mod = self._module(m)
        frame = new_frame("s1", slots={SLOT_FALLBACK_OPEN: True})
        result = await mod.run(frame)
        self.assertEqual(len(result.slots[SLOT_ALERTS]), 1)
        self.assertEqual(len(result.slots[SLOT_CONTENT_META]), 1)
        self.assertEqual(len(result.slots[SLOT_CONTEXT]), 1)

    async def test_context_empty_when_fallback_closed(self) -> None:
        m = DataSourceManager()
        m.context_source.add_context("背景", priority=1.0)
        mod = self._module(m)
        frame = new_frame("s1")  # fallback 默认 False
        result = await mod.run(frame)
        self.assertEqual(result.slots[SLOT_CONTEXT], [])

    async def test_content_meta_normalized(self) -> None:
        m = DataSourceManager()
        m.content_source._score_threshold = 0.0
        m.content_source.add_content("标题内容")
        mod = self._module(m)
        frame = await mod.run(new_frame("s1"))
        meta = frame.slots[SLOT_CONTENT_META][0]
        self.assertIn("id", meta)
        self.assertEqual(meta["title"], "标题内容")
        self.assertEqual(meta["source"], "content")

    async def test_alerts_preserved_raw(self) -> None:
        m = DataSourceManager()
        m.alert_source.add_alert("紧急", {"severity": "critical"})
        mod = self._module(m)
        frame = await mod.run(new_frame("s1"))
        alert = frame.slots[SLOT_ALERTS][0]
        self.assertEqual(alert["content"], "紧急")
        self.assertEqual(alert["metadata"]["severity"], "critical")

    async def test_slot_and_phase_declared(self) -> None:
        mod = self._module()
        self.assertEqual(mod.slot, "source")
        self.assertEqual(mod.phase, "proactive.source")

    async def test_composes_with_runner_gate_first(self) -> None:
        """gate 先开兜底 → source 携带 context，端到端挂进管线。"""
        from initiative.proactive_phases import ProactivePhaseRunner

        class FakeGate:
            slot = "gate"
            phase = "proactive.gate"

            async def run(self, frame):
                frame.slots[SLOT_FALLBACK_OPEN] = True
                return frame

        m = DataSourceManager()
        m.alert_source.add_alert("链路告警")
        m.context_source.add_context("链路背景", priority=1.0)
        runner = ProactivePhaseRunner(
            [FakeGate(), self._module(m)]
        )
        frame = await runner.run(new_frame("s1"))
        self.assertEqual(len(frame.slots[SLOT_ALERTS]), 1)
        self.assertEqual(len(frame.slots[SLOT_CONTEXT]), 1)


class AckTest(unittest.IsolatedAsyncioTestCase):
    def _module_with_ack(self):
        ack = AsyncMock()
        m = DataSourceManager()
        mod = ProactiveSourceModule(
            ProactiveSource(m), ack_fn=ack
        )
        return mod, ack

    async def test_ack_valid_key(self) -> None:
        mod, ack = self._module_with_ack()
        ok = await mod.ack("feed:item42")
        self.assertTrue(ok)
        ack.assert_called_once_with("feed:item42")

    async def test_ack_invalid_key(self) -> None:
        mod, ack = self._module_with_ack()
        ok = await mod.ack("bad-key-no-colon")
        self.assertFalse(ok)
        ack.assert_not_called()

    async def test_ack_exception_returns_false(self) -> None:
        ack = AsyncMock(side_effect=RuntimeError("mcp down"))
        mod = ProactiveSourceModule(
            ProactiveSource(DataSourceManager()), ack_fn=ack
        )
        self.assertFalse(await mod.ack("feed:x"))

    async def test_ack_not_configured(self) -> None:
        mod = ProactiveSourceModule(ProactiveSource(DataSourceManager()))
        self.assertFalse(await mod.ack("feed:x"))


if __name__ == "__main__":
    unittest.main()
