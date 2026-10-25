"""主动链路提示词组装测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from initiative.proactive_frame import new_frame
from initiative.proactive_prompt import (
    CONTEXT_JSON_LIMIT,
    SLOT_PROMPT_CONTEXT,
    SLOT_PROMPT_SYSTEM,
    PromptData,
    ProactivePromptBuilder,
    ProactivePromptModule,
)


class _MemoryProfile:
    """duck-typing memory profile（三个读取器）。"""

    def __init__(self, self_text="", long_term="", recent=""):
        self._self = self_text
        self._long = long_term
        self._recent = recent

    def read_self(self):
        return self._self

    def read_long_term(self):
        return self._long

    def read_recent_context(self):
        return self._recent


class _BrokenMemory:
    def read_self(self):
        raise RuntimeError("disk error")

    def read_long_term(self):
        raise RuntimeError("disk error")

    def read_recent_context(self):
        raise RuntimeError("disk error")


class BuildSystemPromptTest(unittest.TestCase):
    def test_base_directive(self) -> None:
        prompt = ProactivePromptBuilder().build_system_prompt()
        self.assertIn("主动推送决策", prompt)
        self.assertIn("Alert", prompt)
        self.assertIn("Content", prompt)

    def test_plugin_sections_included(self) -> None:
        prompt = ProactivePromptBuilder().build_system_prompt(
            ["情绪状态: 积极", "昼夜 gate: 夜间"]
        )
        self.assertIn("【插件状态】", prompt)
        self.assertIn("情绪状态: 积极", prompt)
        self.assertIn("昼夜 gate: 夜间", prompt)

    def test_blank_plugin_sections_filtered(self) -> None:
        prompt = ProactivePromptBuilder().build_system_prompt(
            ["", "   ", None]  # type: ignore[list-item]
        )
        self.assertNotIn("【插件状态】", prompt)


class BuildContextSectionsTest(unittest.TestCase):
    def _empty_data(self) -> PromptData:
        return PromptData()

    def test_tick_state_always_present(self) -> None:
        builder = ProactivePromptBuilder()
        sections = builder.build_context_sections(
            {"context_as_fallback_open": False}, self._empty_data()
        )
        names = [n for n, _ in sections]
        self.assertEqual(names[0], "tick_state")
        content = sections[0][1]
        self.assertIn("alert_count=0", content)
        self.assertIn("context_fallback=不允许", content)

    def test_fallback_open_label(self) -> None:
        builder = ProactivePromptBuilder()
        sections = builder.build_context_sections(
            {"context_as_fallback_open": True}, self._empty_data()
        )
        self.assertIn("context_fallback=允许", sections[0][1])

    def test_memory_sections_included(self) -> None:
        builder = ProactivePromptBuilder(
            memory=_MemoryProfile(
                self_text="自我认知", long_term="长期", recent="近期"
            )
        )
        sections = builder.build_context_sections(
            {}, self._empty_data()
        )
        names = [n for n, _ in sections]
        self.assertIn("self_model", names)
        self.assertIn("long_term_memory", names)
        self.assertIn("recent_context", names)

    def test_broken_memory_degrades(self) -> None:
        builder = ProactivePromptBuilder(memory=_BrokenMemory())
        sections = builder.build_context_sections({}, self._empty_data())
        names = [n for n, _ in sections]
        self.assertNotIn("self_model", names)
        # tick_state 仍在
        self.assertEqual(names[0], "tick_state")

    def test_no_memory_no_memory_sections(self) -> None:
        builder = ProactivePromptBuilder(memory=None)
        sections = builder.build_context_sections({}, self._empty_data())
        names = [n for n, _ in sections]
        self.assertNotIn("self_model", names)

    def test_empty_memory_values_skipped(self) -> None:
        builder = ProactivePromptBuilder(memory=_MemoryProfile())
        sections = builder.build_context_sections({}, self._empty_data())
        names = [n for n, _ in sections]
        self.assertNotIn("self_model", names)

    def test_workspace_context_included(self) -> None:
        builder = ProactivePromptBuilder(
            workspace_context_fn=lambda: "白名单: 技术类"
        )
        sections = builder.build_context_sections({}, self._empty_data())
        names = [n for n, _ in sections]
        self.assertIn("workspace_proactive_context", names)
        content = dict(sections)["workspace_proactive_context"]
        self.assertIn("白名单: 技术类", content)
        self.assertIn("规则面板", content)

    def test_workspace_context_truncated(self) -> None:
        builder = ProactivePromptBuilder(
            workspace_context_fn=lambda: "x" * 5000
        )
        text = builder.read_workspace_context()
        # 5000 字符截到 3000 + 头部说明行
        self.assertLess(len(text), 3100 + 100)
        self.assertGreater(len(text), 3000)


class RenderAlertBlockTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(ProactivePromptBuilder.render_alert_block([]), "")

    def test_numbered_listing(self) -> None:
        alerts = [
            {"id": "a1", "title": "磁盘告警", "source": "monitor"},
            {"id": "a2", "content": "服务重启", "source": "systemd"},
        ]
        out = ProactivePromptBuilder.render_alert_block(alerts)
        self.assertIn("【Alerts", out)
        self.assertIn("1. [a1] 磁盘告警", out)
        self.assertIn("2. [a2] 服务重启", out)
        self.assertIn("monitor", out)


class RenderContentBlockTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(
            ProactivePromptBuilder.render_content_block([], {}), ""
        )

    def test_body_availability_flag(self) -> None:
        meta = [
            {"id": "c1", "title": "有正文", "source": "rss"},
            {"id": "c2", "title": "无正文", "source": "rss"},
        ]
        store = {"c1": "正文内容"}
        out = ProactivePromptBuilder.render_content_block(meta, store)
        self.assertIn("正文已就绪", out)
        self.assertIn("正文待获取", out)
        self.assertIn("【Content 候选", out)


class RenderContextBlockTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(ProactivePromptBuilder.render_context_block([]), "")

    def test_json_serialized_with_note(self) -> None:
        out = ProactivePromptBuilder.render_context_block(
            [{"sleep_prob": 0.8, "time_local": "23:00"}]
        )
        self.assertIn("【背景上下文】", out)
        self.assertIn("sleep_prob", out)
        self.assertIn("_local", out)  # 说明文字提及本地时间

    def test_truncated_to_limit(self) -> None:
        big = [{"k": "v" * 2000} for _ in range(10)]
        out = ProactivePromptBuilder.render_context_block(big)
        # JSON 部分截断到 900
        json_part = out.split("\n", 2)[-1]
        self.assertLessEqual(len(json_part), CONTEXT_JSON_LIMIT + 10)


class BuildRuntimeContextTest(unittest.TestCase):
    def test_sections_joined(self) -> None:
        builder = ProactivePromptBuilder(
            memory=_MemoryProfile(self_text="自我"),
            workspace_context_fn=lambda: "规则",
        )
        data = PromptData(
            alerts=[{"id": "a1", "title": "告警", "source": "m"}],
        )
        text = builder.build_runtime_context({}, data)
        self.assertIn("[tick_state]", text)
        self.assertIn("[self_model]", text)
        self.assertIn("[proactive_alerts]", text)
        self.assertIn("[workspace_proactive_context]", text)


class PromptModuleTest(unittest.IsolatedAsyncioTestCase):
    def _builder(self, **kwargs):
        return ProactivePromptBuilder(**kwargs)

    async def test_writes_both_slots(self) -> None:
        mod = ProactivePromptModule(self._builder())
        frame = await mod.run(new_frame("s1"))
        self.assertIn(SLOT_PROMPT_SYSTEM, frame.slots)
        self.assertIn(SLOT_PROMPT_CONTEXT, frame.slots)
        self.assertIn("主动推送决策", frame.slots[SLOT_PROMPT_SYSTEM])
        self.assertIn("[tick_state]", frame.slots[SLOT_PROMPT_CONTEXT])

    async def test_reads_data_slots(self) -> None:
        mod = ProactivePromptModule(self._builder())
        frame = new_frame("s1", slots={
            "proactive:alerts": [{"id": "a1", "title": "t", "source": "s"}],
            "proactive:content_meta": [{"id": "c1", "title": "x", "source": "y"}],
            "proactive:content_store": {"c1": "body"},
            "proactive:context": [{"z": 1}],
        })
        result = await mod.run(frame)
        ctx = result.slots[SLOT_PROMPT_CONTEXT]
        self.assertIn("alert_count=1", ctx)
        self.assertIn("[a1] t", ctx)
        self.assertIn("正文已就绪", ctx)

    async def test_plugin_sections_fn(self) -> None:
        mod = ProactivePromptModule(
            self._builder(),
            plugin_sections_fn=lambda: ["情绪: 积极"],
        )
        frame = await mod.run(new_frame("s1"))
        self.assertIn("情绪: 积极", frame.slots[SLOT_PROMPT_SYSTEM])

    async def test_slot_and_phase_declared(self) -> None:
        mod = ProactivePromptModule(self._builder())
        self.assertEqual(mod.slot, "prompt")
        self.assertEqual(mod.phase, "proactive.prompt")

    async def test_composes_with_runner(self) -> None:
        """挂进阶段管线：source 模块先写数据 → prompt 模块读到。"""
        from initiative.proactive_phases import ProactivePhaseRunner

        class FakeSource:
            slot = "source"
            phase = "proactive.source"

            async def run(self, frame):
                frame.slots["proactive:alerts"] = [
                    {"id": "x1", "title": "数据告警", "source": "src"}
                ]
                return frame

        runner = ProactivePhaseRunner(
            [FakeSource(), ProactivePromptModule(self._builder())]
        )
        frame = await runner.run(new_frame("s1"))
        self.assertIn("[x1] 数据告警", frame.slots[SLOT_PROMPT_CONTEXT])


if __name__ == "__main__":
    unittest.main()
