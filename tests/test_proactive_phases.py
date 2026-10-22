"""主动链路阶段管线测试：拓扑排序 + 依赖树 + Runner 链式执行。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from initiative.proactive_frame import ProactiveFrame, new_frame
from initiative.proactive_phases import (
    PROACTIVE_PHASES,
    ProactivePhaseRunner,
    inspect_modules,
    render_dependency_tree,
    sort_modules,
)


@dataclass
class FakeModule:
    """测试用模块：记录执行 + 可选依赖声明。"""

    slot: str
    phase: str = "proactive.gate"
    requires: tuple[str, ...] = ()
    builtin: bool = False
    executed: list[str] = field(default_factory=list)
    slot_to_write: str = ""
    value_to_write: Any = None

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        self.executed.append(self.slot)
        if self.slot_to_write:
            frame.slots[self.slot_to_write] = self.value_to_write
        return frame


class SortModulesTest(unittest.TestCase):
    def test_preserves_declaration_order_no_deps(self) -> None:
        a, b, c = FakeModule("a"), FakeModule("b"), FakeModule("c")
        self.assertEqual(
            [m.slot for m in sort_modules([a, b, c])], ["a", "b", "c"]
        )

    def test_dependency_reorders(self) -> None:
        # c 依赖 a、b → a b c
        c = FakeModule("c", requires=("a", "b"))
        a = FakeModule("a")
        b = FakeModule("b")
        ordered = sort_modules([c, a, b])
        self.assertEqual([m.slot for m in ordered], ["a", "b", "c"])

    def test_unknown_requires_module_disabled(self) -> None:
        # x 依赖不存在的 ghost → x 被剔除
        x = FakeModule("x", requires=("ghost",))
        y = FakeModule("y")
        ordered = sort_modules([x, y])
        self.assertEqual([m.slot for m in ordered], ["y"])

    def test_transitive_disable(self) -> None:
        # x 依赖 ghost（被禁），z 依赖 x（传递禁用）
        x = FakeModule("x", requires=("ghost",))
        z = FakeModule("z", requires=("x",))
        ok = FakeModule("ok")
        ordered = sort_modules([x, z, ok])
        self.assertEqual([m.slot for m in ordered], ["ok"])

    def test_builtin_not_disabled_by_missing_dep(self) -> None:
        b = FakeModule("core", requires=("ghost",), builtin=True)
        out = sort_modules([b])
        self.assertEqual([m.slot for m in out], ["core"])

    def test_builtin_sorts_first(self) -> None:
        normal = FakeModule("normal")
        builtin = FakeModule("builtin_mod", builtin=True)
        ordered = sort_modules([normal, builtin])
        self.assertEqual(
            [m.slot for m in ordered], ["builtin_mod", "normal"]
        )

    def test_duplicate_slot_raises(self) -> None:
        a1 = FakeModule("dup")
        a2 = FakeModule("dup")
        with self.assertRaises(RuntimeError) as ctx:
            sort_modules([a1, a2])
        self.assertIn("重复", str(ctx.exception))

    def test_missing_slot_raises(self) -> None:
        class NoSlot:
            async def run(self, frame):
                return frame

        with self.assertRaises(RuntimeError) as ctx:
            sort_modules([NoSlot()])
        self.assertIn("slot", str(ctx.exception))

    def test_cycle_raises(self) -> None:
        p = FakeModule("p", requires=("q",))
        q = FakeModule("q", requires=("p",))
        with self.assertRaises(RuntimeError) as ctx:
            sort_modules([p, q])
        self.assertIn("循环依赖", str(ctx.exception))

    def test_mixed_known_unknown_requires_disables(self) -> None:
        # 依赖列表混合已知与未知：任一缺失即禁用（不静默丢弃依赖）
        m = FakeModule("m", requires=("ghost", "base"))
        base = FakeModule("base")
        ordered = sort_modules([m, base])
        self.assertEqual([x.slot for x in ordered], ["base"])


class RenderDependencyTreeTest(unittest.TestCase):
    def test_flat_chain(self) -> None:
        a = FakeModule("a")
        b = FakeModule("b", requires=("a",))
        tree = render_dependency_tree([a, b])
        self.assertIn("a", tree)
        self.assertIn("b", tree)

    def test_empty(self) -> None:
        self.assertEqual(render_dependency_tree([]), "")


class InspectModulesTest(unittest.TestCase):
    def test_contains_order_and_tree(self) -> None:
        a = FakeModule("alpha")
        b = FakeModule("beta", requires=("alpha",))
        text = inspect_modules([b, a])
        self.assertIn("执行顺序", text)
        self.assertIn("依赖树", text)
        self.assertIn("alpha", text)
        self.assertIn("beta", text)


class ProactivePhaseRunnerTest(unittest.IsolatedAsyncioTestCase):
    def test_missing_phase_raises(self) -> None:
        class NoPhase:
            async def run(self, frame):
                return frame

        with self.assertRaises(RuntimeError) as ctx:
            ProactivePhaseRunner([NoPhase()])
        self.assertIn("phase", str(ctx.exception))

    def test_unknown_phase_raises(self) -> None:
        m = FakeModule("x", phase="proactive.bogus")
        with self.assertRaises(RuntimeError) as ctx:
            ProactivePhaseRunner([m])
        self.assertIn("未知", str(ctx.exception))

    def test_phases_execute_in_canonical_order(self) -> None:
        # 故意乱序传入：deliver 在 gate 前
        log: list[str] = []

        class Ordered(FakeModule):
            async def run(self, frame):
                log.append(self.slot)
                return frame

        gate = Ordered("gate_mod", phase="proactive.gate")
        deliver = Ordered("deliver_mod", phase="proactive.deliver")
        tick = Ordered("tick_mod", phase="proactive.tick")

        runner = ProactivePhaseRunner([deliver, gate, tick])
        frame = runner._modules_by_phase  # 触发构建
        self.assertIn("proactive.tick", PROACTIVE_PHASES)

        # 阶段顺序：tick < gate < deliver（索引比较）
        self.assertLess(
            PROACTIVE_PHASES.index("proactive.tick"),
            PROACTIVE_PHASES.index("proactive.gate"),
        )
        self.assertLess(
            PROACTIVE_PHASES.index("proactive.gate"),
            PROACTIVE_PHASES.index("proactive.deliver"),
        )

    async def test_run_chains_modules(self) -> None:
        tick = FakeModule(
            "tick", phase="proactive.tick",
            slot_to_write="base_score", value_to_write=0.7,
        )
        gate = FakeModule("gate", phase="proactive.gate")
        runner = ProactivePhaseRunner([tick, gate])
        frame = await runner.run(new_frame("s1"))
        # tick 先执行 → gate 能看到它写的 slot
        self.assertEqual(frame.slots["base_score"], 0.7)
        self.assertEqual(tick.executed, ["tick"])
        self.assertEqual(gate.executed, ["gate"])

    async def test_run_with_no_modules_returns_frame(self) -> None:
        runner = ProactivePhaseRunner([])
        frame = await runner.run(new_frame("s1"))
        self.assertEqual(frame.slots, {})

    async def test_module_can_modify_frame(self) -> None:
        writer = FakeModule(
            "writer", phase="proactive.prompt",
            slot_to_write="prompt_section", value_to_write="问候语",
        )
        runner = ProactivePhaseRunner([writer])
        frame = await runner.run(new_frame("s1"))
        self.assertEqual(frame.slots["prompt_section"], "问候语")

    async def test_intra_phase_dependency_order(self) -> None:
        # 同一 phase 内：late 依赖 early
        late = FakeModule("late", phase="proactive.judge", requires=("early",))
        early = FakeModule("early", phase="proactive.judge")
        runner = ProactivePhaseRunner([late, early])
        await runner.run(new_frame("s1"))
        self.assertEqual(early.executed, ["early"])
        self.assertEqual(late.executed, ["late"])
        # early 先于 late（都只有一次执行，顺序由 sort 保证）
        slots = runner.all_slots()["proactive.judge"]
        self.assertEqual(slots, ["early", "late"])

    def test_modules_by_phase_copy(self) -> None:
        m = FakeModule("m", phase="proactive.gate")
        runner = ProactivePhaseRunner([m])
        by_phase = runner.modules_by_phase
        by_phase["proactive.gate"].append("tamper")
        # 返回的是拷贝，内部不受影响
        self.assertEqual(len(runner.modules_by_phase["proactive.gate"]), 1)

    def test_inspect_lists_only_nonempty_phases(self) -> None:
        m = FakeModule("only_gate", phase="proactive.gate")
        runner = ProactivePhaseRunner([m])
        text = runner.inspect()
        self.assertIn("[proactive.gate]", text)
        self.assertNotIn("[proactive.deliver]", text)

    def test_all_slots(self) -> None:
        a = FakeModule("a", phase="proactive.tick")
        b = FakeModule("b", phase="proactive.tick")
        runner = ProactivePhaseRunner([a, b])
        self.assertEqual(runner.all_slots()["proactive.tick"], ["a", "b"])
        self.assertEqual(runner.all_slots()["proactive.deliver"], [])


class ProactivePhasesConstantTest(unittest.TestCase):
    def test_nine_phases_in_order(self) -> None:
        self.assertEqual(PROACTIVE_PHASES, (
            "proactive.tick",
            "proactive.gate",
            "proactive.source",
            "proactive.drift",
            "proactive.prompt",
            "proactive.judge",
            "proactive.resolve",
            "proactive.deliver",
            "proactive.schedule",
        ))

    def test_unique_phases(self) -> None:
        self.assertEqual(len(PROACTIVE_PHASES), len(set(PROACTIVE_PHASES)))


if __name__ == "__main__":
    unittest.main()
