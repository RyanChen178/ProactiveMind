"""接线审计测试：验证 embeddings / Self 模型 / 三路数据源真实接入。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mind.config import PromptConfig
from mind.prompt import PromptBuilder


def _run(coro):
    return asyncio.run(coro)


class PromptSelfInjectionTest(unittest.TestCase):
    """PromptBuilder 的 self_text 注入。"""

    def test_self_text_block_included(self) -> None:
        builder = PromptBuilder(PromptConfig())
        text = builder.build(
            memory_text="- 记忆事实",
            self_text="# 偏好\n- 喜欢简洁",
        )
        self.assertIn("## 自我认知", text)
        self.assertIn("喜欢简洁", text)
        self.assertIn("## 已有记忆", text)

    def test_empty_self_omitted(self) -> None:
        builder = PromptBuilder(PromptConfig())
        text = builder.build(memory_text="- x", self_text="  ")
        self.assertNotIn("## 自我认知", text)

    def test_backward_compatible_signature(self) -> None:
        builder = PromptBuilder(PromptConfig())
        text = builder.build("- 仅记忆")
        self.assertIn("## 已有记忆", text)
        self.assertNotIn("## 自我认知", text)


class WanderSelfModelTest(unittest.IsolatedAsyncioTestCase):
    """WanderLoop 执行成功后更新 Self.md。"""

    async def test_executed_skill_updates_self_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            skills_dir = workspace / "playbooks"
            skill_dir = skills_dir / "audit-memory"
            skill_dir.mkdir(parents=True)
            (skill_dir / "PLAYBOOK.md").write_text(
                "# 审计\n\n内容", encoding="utf-8",
            )

            from initiative.self_model import SelfModelManager

            self_model = SelfModelManager(workspace)
            self_model.load()

            response = MagicMock(content="掌握了记忆审计能力", usage={})
            provider = MagicMock()
            provider.chat = AsyncMock(return_value=response)

            from initiative.drift import WanderLoop

            wander = WanderLoop(provider, skills_dir, self_model=self_model)
            skills = wander.scan_playbooks()
            self.assertEqual(len(skills), 1)

            result = await wander._execute_skill(skills[0])
            self.assertEqual(result.action, "executed")

            # Self.md 应被写入能力条目
            self_file = workspace / "Self.md"
            self.assertTrue(self_file.exists())
            content = self_file.read_text(encoding="utf-8")
            self.assertIn("掌握了记忆审计能力", content)

    async def test_without_self_model_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skills_dir = Path(temp_dir) / "playbooks"
            skill_dir = skills_dir / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "PLAYBOOK.md").write_text("# 演示\n", encoding="utf-8")

            provider = MagicMock()
            provider.chat = AsyncMock(return_value=MagicMock(content="done", usage={}))

            from initiative.drift import WanderLoop

            wander = WanderLoop(provider, skills_dir, self_model=None)
            skills = wander.scan_playbooks()
            result = await wander._execute_skill(skills[0])
            self.assertEqual(result.action, "executed")
            # 不应创建 Self.md
            self.assertFalse((Path(temp_dir) / "Self.md").exists())


class MindLoopEmbeddingWiringTest(unittest.IsolatedAsyncioTestCase):
    """MindLoop 构造后 MemoryStore 挂载了 EmbeddingStore。"""

    def _make_agent(self, workspace: Path):
        from mind.loop import MindLoop

        agent = MindLoop.__new__(MindLoop)
        agent._config = MagicMock()
        agent._config.workspace = workspace
        return agent

    def test_full_init_attaches_embedding_store(self) -> None:
        """复刻 __init__ 的 embeddings 段，验证挂载路径可用。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "MEMORY.md").write_text(
                "# 长期记忆\n\n- 用户喜欢喝咖啡\n", encoding="utf-8",
            )
            (workspace / "PENDING.md").write_text(
                "# 待归档\n\n- 项目使用 Python\n", encoding="utf-8",
            )

            from mind.memory import MemoryStore

            memory = MemoryStore(workspace)
            self.assertIsNone(memory._embedding_store)  # type: ignore[attr-defined]

            from mind.embeddings import EmbeddingStore, LocalTFIDFBackend

            corpus = [
                line[2:].strip()
                for line in memory.read_all().splitlines()
                if line.strip().startswith("- ")
            ]
            corpus.extend(memory.read_pending())

            backend = LocalTFIDFBackend()
            backend.fit(corpus)
            store = EmbeddingStore(backend, workspace / "embeddings.db")
            try:
                memory.attach_embedding_store(store)
                self.assertIsNotNone(memory._embedding_store)  # type: ignore[attr-defined]
                # semantic_recall 不再降级：返回带分数的语义结果
                results = memory.semantic_recall("咖啡", top_k=1)
                self.assertTrue(any("咖啡" in f for f, _ in results))
            finally:
                store.close()

    def test_semantic_recall_without_store_degrades(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            from mind.memory import MemoryStore

            memory = MemoryStore(Path(temp_dir))
            memory.append("hello keyword")
            results = memory.semantic_recall("hello", top_k=3)
            # 降级路径：score 固定 1.0
            self.assertEqual(results, [("hello keyword", 1.0)])


class DataSourceWiringTest(unittest.TestCase):
    """_build_agent 把 DataSourceManager 传给 InitiativeLoop。"""

    def test_build_agent_passes_data_sources(self) -> None:
        """静态检查 app.py 源码中 InitiativeLoop 构造含 data_source_manager。"""
        import inspect

        import app

        src = inspect.getsource(app._build_agent)
        self.assertIn("DataSourceManager", src)
        self.assertIn("data_source_manager=", src)


class InitiativePushIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """三路数据源 + InitiativeLoop 推送链路（带 alert）。"""

    async def test_alert_reaches_push_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            from initiative.data_sources import DataSourceManager
            from initiative.loop import InitiativeLoop
            from initiative.presence import PresenceStore
            from datetime import datetime, timedelta, timezone

            presence = PresenceStore(Path(temp_dir) / "presence.db")
            presence.record_user_message(
                datetime.now(timezone.utc) - timedelta(hours=6)
            )
            pushed: list[str] = []

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()
            manager.alert_source.add_alert("磁盘空间不足")
            loop = InitiativeLoop(
                presence,
                data_source_manager=manager,
                push_callback=push_callback,
            )
            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "pushed")
            self.assertEqual(result.pushed_source, "alert")
            self.assertEqual(len(pushed), 1)
            self.assertIn("磁盘空间不足", pushed[0])


if __name__ == "__main__":
    unittest.main()