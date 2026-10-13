"""主动推送 Gate 准入链与 TickContext 测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from initiative.proactive_gate import GateResult, ProactiveGateChain
from initiative.proactive_tick_context import AgentTickContext, new_tick_id


def _make_cfg(**overrides):
    base = {
        "default_chat_id": "u1",
        "default_channel": "web",
        "agent_tick_delivery_cooldown_hours": 1,
        "agent_tick_context_prob": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_state_store(
    *,
    deliveries_in_window: int = 0,
    last_context_only_at: datetime | None = None,
    context_only_in_window: int = 0,
):
    store = MagicMock()
    store.count_deliveries_in_window = MagicMock(return_value=deliveries_in_window)
    store.get_last_context_only_at = MagicMock(return_value=last_context_only_at)
    store.count_context_only_in_window = MagicMock(return_value=context_only_in_window)
    return store


class AgentTickContextTest(unittest.TestCase):
    def test_new_tick_id_is_short(self) -> None:
        tid = new_tick_id()
        self.assertEqual(len(tid), 8)

    def test_new_tick_id_unique(self) -> None:
        ids = {new_tick_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_defaults(self) -> None:
        ctx = AgentTickContext()
        self.assertEqual(ctx.session_key, "")
        self.assertFalse(ctx.context_as_fallback_open)
        self.assertIsNotNone(ctx.now_utc)
        self.assertEqual(len(ctx.tick_id), 8)

    def test_now_utc_close_to_real_now(self) -> None:
        before = datetime.now(timezone.utc)
        ctx = AgentTickContext()
        after = datetime.now(timezone.utc)
        self.assertGreaterEqual(ctx.now_utc, before - timedelta(seconds=1))
        self.assertLessEqual(ctx.now_utc, after + timedelta(seconds=1))


class GateNoTargetTest(unittest.TestCase):
    def test_no_chat_id_blocks(self) -> None:
        cfg = _make_cfg(default_chat_id="")
        store = _make_state_store()
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "no_target")


class GateBusyTest(unittest.TestCase):
    def test_passive_busy_blocks(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            passive_busy_fn=lambda k: True,
        )
        result = chain.check(AgentTickContext())
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "busy")

    def test_not_busy_passes_busy_gate(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            passive_busy_fn=lambda k: False,
        )
        result = chain.check(AgentTickContext())
        self.assertFalse(result.blocked)
        self.assertEqual(result.reason, "passed")


class GateCooldownTest(unittest.TestCase):
    def test_recent_delivery_blocks(self) -> None:
        cfg = _make_cfg(agent_tick_delivery_cooldown_hours=1)
        store = _make_state_store(deliveries_in_window=1)
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "cooldown")

    def test_no_recent_delivery_passes_cooldown(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store(deliveries_in_window=0)
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertFalse(result.blocked)


class GateAnyactionTest(unittest.TestCase):
    def _anyaction(self, should_act: bool, meta=None):
        gate = MagicMock()
        gate.should_act = MagicMock(return_value=(should_act, meta))
        return gate

    def test_anyaction_blocks_when_should_act_false(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            any_action_gate=self._anyaction(False, "max_reached"),
        )
        result = chain.check(AgentTickContext())
        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "presence")

    def test_anyaction_passes_when_should_act_true(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            any_action_gate=self._anyaction(True),
        )
        result = chain.check(AgentTickContext())
        self.assertFalse(result.blocked)

    def test_anyaction_exception_fails_open(self) -> None:
        cfg = _make_cfg()
        store = _make_state_store()
        gate = MagicMock()
        gate.should_act = MagicMock(side_effect=RuntimeError("down"))
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            any_action_gate=gate,
        )
        result = chain.check(AgentTickContext())
        self.assertFalse(result.blocked)


class GateFallbackTest(unittest.TestCase):
    def test_fallback_prob_zero(self) -> None:
        cfg = _make_cfg(agent_tick_context_prob=0.0)
        store = _make_state_store()
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertFalse(result.context_as_fallback_open)

    def test_fallback_prob_one_always_open(self) -> None:
        cfg = _make_cfg(agent_tick_context_prob=1.0)
        store = _make_state_store()
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertTrue(result.context_as_fallback_open)

    def test_fallback_rng_zero(self) -> None:
        cfg = _make_cfg(agent_tick_context_prob=0.5)
        store = _make_state_store()
        rng = SimpleNamespace(random=lambda: 0.0)
        chain = ProactiveGateChain(
            cfg=cfg, session_key="s1", state_store=store, rng=rng,
        )
        result = chain.check(AgentTickContext())
        self.assertTrue(result.context_as_fallback_open)

    def test_fallback_rng_one(self) -> None:
        cfg = _make_cfg(agent_tick_context_prob=0.5)
        store = _make_state_store()
        rng = SimpleNamespace(random=lambda: 0.9)
        chain = ProactiveGateChain(
            cfg=cfg, session_key="s1", state_store=store, rng=rng,
        )
        result = chain.check(AgentTickContext())
        self.assertFalse(result.context_as_fallback_open)


class GateStateStoreFailureTest(unittest.TestCase):
    def test_count_failure_fails_open(self) -> None:
        cfg = _make_cfg()
        store = MagicMock()
        store.count_deliveries_in_window = MagicMock(
            side_effect=RuntimeError("db down")
        )
        store.get_last_context_only_at = MagicMock(return_value=None)
        store.count_context_only_in_window = MagicMock(return_value=0)
        chain = ProactiveGateChain(cfg=cfg, session_key="s1", state_store=store)
        result = chain.check(AgentTickContext())
        self.assertFalse(result.blocked)


class GateOrderingTest(unittest.TestCase):
    def test_no_target_takes_precedence(self) -> None:
        cfg = _make_cfg(default_chat_id="")
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            passive_busy_fn=lambda k: True,
        )
        result = chain.check(AgentTickContext())
        self.assertEqual(result.reason, "no_target")

    def test_busy_after_no_target(self) -> None:
        cfg = _make_cfg(default_chat_id="u1")
        store = _make_state_store()
        chain = ProactiveGateChain(
            cfg=cfg,
            session_key="s1",
            state_store=store,
            passive_busy_fn=lambda k: True,
        )
        result = chain.check(AgentTickContext())
        self.assertEqual(result.reason, "busy")


if __name__ == "__main__":
    unittest.main()
