"""主动链路消解模块测试：投递指纹 + 三段决策 + ack 分级。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from initiative.proactive_frame import new_frame
from initiative.proactive_resolve import (
    ACTION_SEND,
    ACTION_SKIP,
    CITED_ACK_TTL_HOURS,
    DISCARDED_ACK_TTL_HOURS,
    UNCITED_ACK_TTL_HOURS,
    ProactiveResolveModule,
    ProactiveResolver,
    ResolveDecision,
    ack_ttl_for,
    build_ack_plan,
    build_delivery_key,
    normalize_url,
)


def _make_deduper(is_dup: bool, reason: str = "雷同"):
    d = MagicMock()
    d.is_duplicate = AsyncMock(return_value=(is_dup, reason))
    return d


class NormalizeUrlTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(normalize_url(""), "")

    def test_lowercase_scheme_host(self) -> None:
        self.assertEqual(
            normalize_url("HTTPS://Example.COM/path"),
            "https://example.com/path",
        )

    def test_trailing_slash_stripped(self) -> None:
        self.assertEqual(
            normalize_url("https://e.com/path/"),
            "https://e.com/path",
        )

    def test_fragment_dropped_query_kept(self) -> None:
        self.assertEqual(
            normalize_url("https://e.com/p?x=1#frag"),
            "https://e.com/p?x=1",
        )


class BuildDeliveryKeyTest(unittest.TestCase):
    def test_url_ref_preferred(self) -> None:
        k1 = build_delivery_key(
            cited_ids=["c1"],
            content_meta=[{"id": "c1", "url": "https://e.com/a",
                           "title": "t", "source": "s"}],
            final_message="m",
        )
        k2 = build_delivery_key(
            cited_ids=["c1"],
            content_meta=[{"id": "c1", "url": "https://e.com/a/",
                           "title": "different", "source": "x"}],
            final_message="different message",
        )
        # URL 相同（归一化后）→ 同指纹
        self.assertEqual(k1, k2)

    def test_title_ref_when_no_url(self) -> None:
        k1 = build_delivery_key(
            cited_ids=["c1"],
            content_meta=[{"id": "c1", "source": "rss", "title": "标题"}],
            final_message="",
        )
        k2 = build_delivery_key(
            cited_ids=["c1"],
            content_meta=[{"id": "c1", "source": "RSS", "title": "标题"}],
            final_message="",
        )
        # source 大小写归一 → 同指纹
        self.assertEqual(k1, k2)

    def test_unknown_cited_falls_to_id(self) -> None:
        key = build_delivery_key(
            cited_ids=["ghost"], content_meta=[], final_message="m",
        )
        self.assertEqual(len(key), 16)

    def test_no_cites_uses_message_prefix(self) -> None:
        k1 = build_delivery_key(
            cited_ids=[], content_meta=[], final_message="x" * 800,
        )
        k2 = build_delivery_key(
            cited_ids=[], content_meta=[],
            final_message="x" * 800 + "尾部不同不影响前 500",
        )
        # 前 500 相同 → 同指纹
        self.assertEqual(k1, k2)

    def test_key_is_16_hex(self) -> None:
        key = build_delivery_key(
            cited_ids=["a"], content_meta=[], final_message="m",
        )
        self.assertEqual(len(key), 16)
        int(key, 16)  # 合法 hex


class ResolverDecisionTest(unittest.IsolatedAsyncioTestCase):
    async def test_non_reply_skips(self) -> None:
        r = ProactiveResolver()
        d = await r.resolve(
            terminal_action="skip", final_message="", skip_reason="no_content",
        )
        self.assertEqual(d.action, ACTION_SKIP)
        self.assertEqual(d.reason, "no_content")

    async def test_non_reply_default_reason(self) -> None:
        r = ProactiveResolver()
        d = await r.resolve(terminal_action="", final_message="")
        self.assertEqual(d.reason, "no_content")

    async def test_delivery_dedupe_skips(self) -> None:
        r = ProactiveResolver(
            is_delivery_duplicate_fn=lambda key, hours: True,
        )
        d = await r.resolve(
            terminal_action="reply", final_message="消息",
            cited_ids=["c1"],
        )
        self.assertEqual(d.action, ACTION_SKIP)
        self.assertEqual(d.reason, "delivery_dedupe")
        self.assertEqual(len(d.delivery_key), 16)

    async def test_message_dedupe_skips(self) -> None:
        r = ProactiveResolver(
            deduper=_make_deduper(True, "语义雷同"),
        )
        d = await r.resolve(
            terminal_action="reply", final_message="消息",
        )
        self.assertEqual(d.action, ACTION_SKIP)
        self.assertEqual(d.reason, "message_dedupe")

    async def test_all_pass_sends(self) -> None:
        r = ProactiveResolver(
            is_delivery_duplicate_fn=lambda key, hours: False,
            deduper=_make_deduper(False),
        )
        d = await r.resolve(
            terminal_action="reply", final_message="新消息",
        )
        self.assertEqual(d.action, ACTION_SEND)
        self.assertEqual(d.reason, "passed")
        self.assertEqual(d.message, "新消息")

    async def test_message_dedupe_disabled(self) -> None:
        deduper = _make_deduper(True)  # 判重但被开关关闭
        r = ProactiveResolver(
            deduper=deduper, message_dedupe_enabled=False,
        )
        d = await r.resolve(
            terminal_action="reply", final_message="消息",
        )
        self.assertEqual(d.action, ACTION_SEND)
        deduper.is_duplicate.assert_not_called()

    async def test_delivery_dedupe_zero_hours_skips_check(self) -> None:
        dup_fn = MagicMock(return_value=True)
        r = ProactiveResolver(
            is_delivery_duplicate_fn=dup_fn,
            delivery_dedupe_hours=0,
        )
        d = await r.resolve(terminal_action="reply", final_message="m")
        self.assertEqual(d.action, ACTION_SEND)
        dup_fn.assert_not_called()

    async def test_delivery_check_exception_fails_open(self) -> None:
        r = ProactiveResolver(
            is_delivery_duplicate_fn=MagicMock(
                side_effect=RuntimeError("db down")
            ),
        )
        d = await r.resolve(terminal_action="reply", final_message="m")
        self.assertEqual(d.action, ACTION_SEND)

    async def test_deduper_exception_fails_open(self) -> None:
        broken = MagicMock()
        broken.is_duplicate = AsyncMock(side_effect=RuntimeError("llm down"))
        r = ProactiveResolver(deduper=broken)
        d = await r.resolve(terminal_action="reply", final_message="m")
        self.assertEqual(d.action, ACTION_SEND)

    async def test_delivery_dedupe_before_message_dedupe(self) -> None:
        """投递级重复优先命中，不再调消息去重。"""
        deduper = _make_deduper(True)
        r = ProactiveResolver(
            deduper=deduper,
            is_delivery_duplicate_fn=lambda key, hours: True,
        )
        d = await r.resolve(terminal_action="reply", final_message="m")
        self.assertEqual(d.reason, "delivery_dedupe")
        deduper.is_duplicate.assert_not_called()


class AckTtlTest(unittest.TestCase):
    def test_cited_longest(self) -> None:
        self.assertEqual(
            ack_ttl_for(
                "a",
                cited_ids={"a"}, interesting_ids=set(), discarded_ids=set(),
            ),
            CITED_ACK_TTL_HOURS,
        )

    def test_discarded_longest_suppression(self) -> None:
        self.assertEqual(
            ack_ttl_for(
                "b",
                cited_ids=set(), interesting_ids=set(), discarded_ids={"b"},
            ),
            DISCARDED_ACK_TTL_HOURS,
        )

    def test_uncited_interesting_short(self) -> None:
        self.assertEqual(
            ack_ttl_for(
                "c",
                cited_ids=set(), interesting_ids={"c"}, discarded_ids=set(),
            ),
            UNCITED_ACK_TTL_HOURS,
        )

    def test_cited_wins_over_discarded(self) -> None:
        # 同 id 在多个集合：cited 优先
        self.assertEqual(
            ack_ttl_for(
                "x",
                cited_ids={"x"}, interesting_ids=set(), discarded_ids={"x"},
            ),
            CITED_ACK_TTL_HOURS,
        )

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(
            ack_ttl_for(
                "z", cited_ids=set(), interesting_ids=set(), discarded_ids=set(),
            )
        )

    def test_build_ack_plan_sorted_and_complete(self) -> None:
        plan = build_ack_plan(
            cited_ids={"b", "a"},
            interesting_ids={"c"},
            discarded_ids={"d"},
        )
        self.assertEqual(
            plan,
            [
                ("a", CITED_ACK_TTL_HOURS),
                ("b", CITED_ACK_TTL_HOURS),
                ("c", UNCITED_ACK_TTL_HOURS),
                ("d", DISCARDED_ACK_TTL_HOURS),
            ],
        )

    def test_build_ack_plan_empty(self) -> None:
        self.assertEqual(
            build_ack_plan(
                cited_ids=set(), interesting_ids=set(), discarded_ids=set(),
            ),
            [],
        )


class ResolveModuleTest(unittest.IsolatedAsyncioTestCase):
    def _module(self, **kwargs):
        return ProactiveResolveModule(ProactiveResolver(**kwargs))

    async def test_send_path_writes_slots(self) -> None:
        mod = self._module()
        frame = new_frame("s1", slots={
            "terminal_action": "reply",
            "final_message": "要发的消息",
            "cited_item_ids": ["c1"],
            "proactive:content_meta": [{"id": "c1", "url": "https://e.com/x"}],
        })
        result = await mod.run(frame)
        self.assertEqual(result.slots["decision"], ACTION_SEND)
        self.assertEqual(result.slots["draft_message"], "要发的消息")
        self.assertEqual(len(result.slots["delivery_key"]), 16)

    async def test_skip_path_writes_reason(self) -> None:
        mod = self._module()
        frame = new_frame("s1", slots={
            "terminal_action": "skip",
            "skip_reason": "no_content",
        })
        result = await mod.run(frame)
        self.assertEqual(result.slots["decision"], ACTION_SKIP)
        self.assertEqual(result.slots["decision_reason"], "no_content")

    async def test_missing_terminal_defaults_skip(self) -> None:
        mod = self._module()
        frame = await mod.run(new_frame("s1"))
        self.assertEqual(frame.slots["decision"], ACTION_SKIP)

    async def test_slot_and_phase_declared(self) -> None:
        mod = self._module()
        self.assertEqual(mod.slot, "resolve")
        self.assertEqual(mod.phase, "proactive.resolve")

    async def test_composes_with_runner_full_chain(self) -> None:
        """judge → resolve → deliver 全链路挂进阶段管线端到端。"""
        from initiative.proactive_deliver import (
            SLOT_DECISION,
            SLOT_DRAFT_MESSAGE,
            ProactiveDeliverModule,
            ProactiveDeliverer,
        )
        from initiative.proactive_phases import ProactivePhaseRunner

        class FakeJudge:
            slot = "judge"
            phase = "proactive.judge"

            async def run(self, frame):
                frame.slots["terminal_action"] = "reply"
                frame.slots["final_message"] = "管线消息"
                return frame

        async def _send(text):
            return True

        deliverer = ProactiveDeliverer(
            session_key="web:u1", send_fn=_send, state_store=None,
        )
        runner = ProactivePhaseRunner([
            FakeJudge(),
            self._module(),
            ProactiveDeliverModule(deliverer),
        ])
        frame = await runner.run(new_frame("s1"))
        # resolve 把 judge 的终态消解成 send → deliver 实际发出
        self.assertEqual(frame.slots[SLOT_DECISION], ACTION_SEND)
        self.assertEqual(frame.slots["delivery_status"], "sent")
        self.assertTrue(frame.slots["delivered"])


if __name__ == "__main__":
    unittest.main()
