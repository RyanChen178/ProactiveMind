"""主动链路投递模块测试：去重闸 + 发送 + 落库 + phase 壳。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock

from initiative.proactive_deliver import (
    DECISION_SEND,
    DECISION_SKIP,
    SLOT_DECISION,
    SLOT_DELIVERY_REASON,
    SLOT_DELIVERY_STATUS,
    SLOT_DELIVERED,
    SLOT_DRAFT_MESSAGE,
    ProactiveDeliverer,
    ProactiveDeliverModule,
)
from initiative.proactive_frame import new_frame
from initiative.proactive_state import ProactiveStateStore


class _StateStoreFixture:
    def __init__(self):
        self._tmp = TemporaryDirectory()
        self.store = ProactiveStateStore(Path(self._tmp.name) / "st.db")

    def cleanup(self):
        self.store.close()
        self._tmp.cleanup()


def _make_deduper(is_dup: bool, reason: str = "雷同"):
    deduper = MagicMock()
    deduper.is_duplicate = AsyncMock(return_value=(is_dup, reason))
    return deduper


class DelivererPipelineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fixture = _StateStoreFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def _deliverer(self, send_fn=None, deduper=None, **kwargs):
        defaults = dict(
            session_key="web:u1",
            send_fn=send_fn,
            state_store=self.fixture.store,
            deduper=deduper,
        )
        defaults.update(kwargs)
        return ProactiveDeliverer(**defaults)

    async def test_success_records_delivery(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(send_fn=send)
        result = await d.deliver("重要通知", message_id="m1")
        self.assertTrue(result["delivered"])
        self.assertEqual(result["status"], "sent")
        # 落库可查
        msgs = self.fixture.store.recent_delivery_messages("web:u1")
        self.assertIn("m1", msgs)

    async def test_duplicate_blocked(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(
            send_fn=send, deduper=_make_deduper(True, "语义雷同")
        )
        result = await d.deliver("重复消息")
        self.assertFalse(result["delivered"])
        self.assertEqual(result["status"], "deduped")
        self.assertEqual(result["reason"], "语义雷同")
        # 未发送、未落库
        send.assert_not_called()
        self.assertEqual(
            self.fixture.store.recent_delivery_messages("web:u1"), []
        )

    async def test_deduper_passes_then_sends(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(
            send_fn=send, deduper=_make_deduper(False, "无关")
        )
        result = await d.deliver("新消息")
        self.assertTrue(result["delivered"])
        send.assert_called_once_with("新消息")

    async def test_send_failure_not_recorded(self) -> None:
        send = AsyncMock(return_value=False)
        d = self._deliverer(send_fn=send)
        result = await d.deliver("会失败的消息")
        self.assertFalse(result["delivered"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            self.fixture.store.recent_delivery_messages("web:u1"), []
        )

    async def test_send_exception_not_recorded(self) -> None:
        async def _boom(text):
            raise RuntimeError("channel down")

        d = self._deliverer(send_fn=_boom)
        result = await d.deliver("异常路径")
        self.assertFalse(result["delivered"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            self.fixture.store.recent_delivery_messages("web:u1"), []
        )

    async def test_deduper_exception_fails_open(self) -> None:
        broken = MagicMock()
        broken.is_duplicate = AsyncMock(side_effect=RuntimeError("llm down"))
        send = AsyncMock(return_value=True)
        d = self._deliverer(send_fn=send, deduper=broken)
        result = await d.deliver("去重挂了也要能发")
        self.assertTrue(result["delivered"])

    async def test_empty_message_rejected(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(send_fn=send)
        result = await d.deliver("   ")
        self.assertFalse(result["delivered"])
        self.assertEqual(result["reason"], "empty message")
        send.assert_not_called()

    async def test_send_fn_missing(self) -> None:
        d = self._deliverer(send_fn=None)
        result = await d.deliver("无处可发")
        self.assertFalse(result["delivered"])
        self.assertEqual(result["reason"], "send_fn not configured")

    async def test_message_id_defaults_to_prefix(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(send_fn=send)
        await d.deliver("没有显式 ID 的消息")
        msgs = self.fixture.store.recent_delivery_messages("web:u1")
        self.assertEqual(len(msgs), 1)
        # 缺省 message_id 取正文前 32 字符
        self.assertEqual(msgs[0][:8], "没有显式 ID 的消息"[:8])

    async def test_payload_kind_and_preview(self) -> None:
        send = AsyncMock(return_value=True)
        d = self._deliverer(send_fn=send)
        await d.deliver(
            "x" * 200, message_id="kind-test", payload_kind="alert"
        )
        row = self.fixture.store._db.execute(
            "SELECT payload_kind, payload_preview FROM deliveries "
            "WHERE message_id = 'kind-test'"
        ).fetchone()
        self.assertEqual(row["payload_kind"], "alert")
        self.assertEqual(len(row["payload_preview"]), 80)

    async def test_recent_proactive_passed_to_deduper(self) -> None:
        deduper = _make_deduper(False)
        recent = [{"content": "旧消息"}]
        d = self._deliverer(
            send_fn=AsyncMock(return_value=True),
            deduper=deduper,
            recent_proactive_fn=lambda: recent,
        )
        await d.deliver("新消息")
        deduper.is_duplicate.assert_called_once_with(
            "新消息", recent_proactive=recent
        )


class DelivererNoStateStoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_works_without_store(self) -> None:
        send = AsyncMock(return_value=True)
        d = ProactiveDeliverer(
            session_key="s", send_fn=send, state_store=None,
        )
        result = await d.deliver("无库也能发")
        self.assertTrue(result["delivered"])


class DeliverModuleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.fixture = _StateStoreFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def _module(self, send_return=True):
        send = AsyncMock(return_value=send_return)
        deliverer = ProactiveDeliverer(
            session_key="web:u1",
            send_fn=send,
            state_store=self.fixture.store,
        )
        return ProactiveDeliverModule(deliverer)

    async def test_send_decision_delivers(self) -> None:
        mod = self._module()
        frame = new_frame("s1", slots={
            SLOT_DECISION: DECISION_SEND,
            SLOT_DRAFT_MESSAGE: "要发的消息",
            "message_id": "mid-1",
        })
        result = await mod.run(frame)
        self.assertTrue(result.slots[SLOT_DELIVERED])
        self.assertEqual(result.slots[SLOT_DELIVERY_STATUS], "sent")

    async def test_skip_decision_skips(self) -> None:
        mod = self._module()
        frame = new_frame("s1", slots={SLOT_DECISION: DECISION_SKIP})
        result = await mod.run(frame)
        self.assertFalse(result.slots[SLOT_DELIVERED])
        self.assertEqual(result.slots[SLOT_DELIVERY_STATUS], "skipped")
        self.assertIn("decision=skip", result.slots[SLOT_DELIVERY_REASON])

    async def test_missing_decision_defaults_skip(self) -> None:
        mod = self._module()
        frame = new_frame("s1")
        result = await mod.run(frame)
        self.assertEqual(result.slots[SLOT_DELIVERY_STATUS], "skipped")

    async def test_send_failure_status(self) -> None:
        mod = self._module(send_return=False)
        frame = new_frame("s1", slots={
            SLOT_DECISION: DECISION_SEND,
            SLOT_DRAFT_MESSAGE: "发不出去",
        })
        result = await mod.run(frame)
        self.assertFalse(result.slots[SLOT_DELIVERED])
        self.assertEqual(result.slots[SLOT_DELIVERY_STATUS], "failed")

    async def test_slot_and_phase_declared(self) -> None:
        mod = self._module()
        self.assertEqual(mod.slot, "deliver")
        self.assertEqual(mod.phase, "proactive.deliver")

    async def test_composes_with_runner(self) -> None:
        """judge 写决策 → deliver 执行，端到端挂进阶段管线。"""
        from initiative.proactive_phases import ProactivePhaseRunner

        class FakeJudge:
            slot = "judge"
            phase = "proactive.judge"

            async def run(self, frame):
                frame.slots[SLOT_DECISION] = DECISION_SEND
                frame.slots[SLOT_DRAFT_MESSAGE] = "管线投递"
                return frame

        runner = ProactivePhaseRunner([FakeJudge(), self._module()])
        frame = await runner.run(new_frame("s1"))
        self.assertTrue(frame.slots[SLOT_DELIVERED])
        self.assertEqual(frame.slots[SLOT_DELIVERY_STATUS], "sent")
        msgs = self.fixture.store.recent_delivery_messages("web:u1")
        self.assertEqual(len(msgs), 1)


if __name__ == "__main__":
    unittest.main()
