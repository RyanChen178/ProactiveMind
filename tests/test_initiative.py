"""主动推送循环测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from initiative.data_sources import (
    AlertDataSource,
    ContentDataSource,
    ContextDataSource,
    DataSourceManager,
)
from initiative.drift import WanderResult
from initiative.loop import InitiativeLoop
from initiative.presence import PresenceStore


class InitiativeLoopTest(unittest.IsolatedAsyncioTestCase):
    def _make_loop(
        self, temp_dir: str, *, busy=False, max_ticks=1
    ) -> tuple[InitiativeLoop, PresenceStore]:
        presence = PresenceStore(Path(temp_dir) / "presence.db")
        presence.record_user_message(
            datetime.now(timezone.utc) - timedelta(hours=6)
        )
        loop = InitiativeLoop(
            presence,
            is_passive_busy=lambda: busy,
            max_ticks=max_ticks,
        )
        return loop, presence

    async def test_skips_when_passive_busy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir, busy=True)

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "skipped")
            self.assertEqual(result.reason, "passive_busy")

    async def test_skips_during_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            presence.record_proactive(datetime.now(timezone.utc))

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "skipped")
            self.assertEqual(result.reason, "cooldown")

    async def test_returns_no_content_when_no_data_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "no_content")
            self.assertGreater(result.urgency, 0.0)
            self.assertGreater(result.interval_s, 60.0)

    async def test_pushes_drift_result_via_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def fake_drift_run():
                return WanderResult(
                    action="executed",
                    skill_name="audit-memory",
                    summary="发现 2 条过时记忆",
                )

            async def push_callback(content: str) -> None:
                pushed.append(content)

            loop._drift_loop = MagicMock()
            loop._drift_loop.run = fake_drift_run
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "pushed")
            self.assertIn("audit-memory", result.pushed_content)
            self.assertIn("发现 2 条过时记忆", result.pushed_content)
            self.assertEqual(len(pushed), 1)

    async def test_falls_back_to_drift_when_no_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)

            async def fake_drift_run():
                return WanderResult(
                    action="executed",
                    skill_name="audit-memory",
                    summary="审计完成",
                )

            loop._drift_loop = MagicMock()
            loop._drift_loop.run = fake_drift_run

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "drift")
            self.assertEqual(result.pushed_content, "")

    async def test_loop_stops_after_max_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir, max_ticks=2)

            original_sleep = asyncio.sleep
            sleep_calls: list[float] = []

            async def fast_sleep(seconds):
                sleep_calls.append(seconds)
                await original_sleep(0)

            asyncio.sleep = fast_sleep  # type: ignore
            try:
                await loop.run()
            finally:
                asyncio.sleep = original_sleep  # type: ignore
                presence.close()

            self.assertEqual(loop._tick_count, 2)
            self.assertGreater(len(sleep_calls), 0)

    async def test_pushes_alert_with_highest_priority(self) -> None:
        """Alert 应优先于其他数据源被推送。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()
            manager.content_source.add_content("普通内容")
            manager.context_source.add_context("上下文", priority=1.0)
            manager.alert_source.add_alert("系统严重错误")

            loop._data_source_manager = manager
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "pushed")
            self.assertEqual(result.pushed_source, "alert")
            self.assertIn("系统严重错误", result.pushed_content)
            self.assertEqual(len(pushed), 1)

    async def test_no_data_source_no_push(self) -> None:
        """数据源为空时不应推送（无 callback 时）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()
            loop._data_source_manager = manager
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "no_content")
            self.assertEqual(result.pushed_source, "")
            self.assertEqual(len(pushed), 0)

    async def test_falls_back_to_drift_when_data_source_empty(self) -> None:
        """数据源为空时回退到 Drift。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def fake_drift_run():
                return WanderResult(
                    action="executed",
                    skill_name="audit-memory",
                    summary="审计完成",
                )

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()  # 空数据源
            loop._data_source_manager = manager
            loop._drift_loop = MagicMock()
            loop._drift_loop.run = fake_drift_run
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            # 数据源为空，drift 执行了 skill，所以应推 drift 结果
            self.assertEqual(result.action, "pushed")
            self.assertEqual(result.pushed_source, "drift")
            self.assertIn("audit-memory", result.pushed_content)
            self.assertEqual(len(pushed), 1)

    async def test_data_source_takes_precedence_over_drift(self) -> None:
        """数据源有内容时优先于 Drift 推送。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def fake_drift_run():
                return WanderResult(
                    action="executed",
                    skill_name="audit-memory",
                    summary="审计完成",
                )

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()
            manager.alert_source.add_alert("紧急告警")

            loop._data_source_manager = manager
            loop._drift_loop = MagicMock()
            loop._drift_loop.run = fake_drift_run
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "pushed")
            self.assertEqual(result.pushed_source, "alert")
            self.assertIn("紧急告警", result.pushed_content)
            self.assertNotIn("audit-memory", result.pushed_content)
            self.assertEqual(len(pushed), 1)

    async def test_push_without_callback_only_logs(self) -> None:
        """无 push_callback 时数据源不应推送（保持原行为）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)

            manager = DataSourceManager()
            manager.alert_source.add_alert("紧急告警")

            loop._data_source_manager = manager
            # 没有 push_callback

            result = await loop._tick()
            presence.close()

            # 没有 callback 时也不进入 Drift（保持向后兼容）
            self.assertEqual(result.action, "no_content")

    async def test_content_source_filtered_by_threshold(self) -> None:
        """Content 数据源评分低于阈值时不应被推送。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def push_callback(content: str) -> None:
                pushed.append(content)

            async def low_scorer(content: str) -> float:
                return 0.3  # 低于默认阈值 0.7

            manager = DataSourceManager()
            manager.content_source._llm_scorer = low_scorer
            manager.content_source.add_content("低分内容")

            loop._data_source_manager = manager
            loop._push_callback = push_callback

            result = await loop._tick()
            presence.close()

            self.assertEqual(result.action, "no_content")
            self.assertEqual(len(pushed), 0)

    async def test_records_proactive_after_data_source_push(self) -> None:
        """数据源推送后应记录主动推送时间。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loop, presence = self._make_loop(temp_dir)
            pushed: list[str] = []

            async def push_callback(content: str) -> None:
                pushed.append(content)

            manager = DataSourceManager()
            manager.alert_source.add_alert("告警")
            loop._data_source_manager = manager
            loop._push_callback = push_callback

            before = presence.get_last_proactive_at()
            await loop._tick()
            after = presence.get_last_proactive_at()
            presence.close()

            self.assertIsNone(before)
            self.assertIsNotNone(after)


class PresenceStoreTest(unittest.TestCase):
    def test_records_and_reads_user_message_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PresenceStore(Path(temp_dir) / "presence.db")
            ts = datetime(2026, 8, 19, 14, 30, 0, tzinfo=timezone.utc)
            store.record_user_message(ts)

            result = store.get_last_user_at()
            self.assertIsNotNone(result)
            self.assertEqual(result, ts)
            store.close()

    def test_returns_none_when_no_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PresenceStore(Path(temp_dir) / "presence.db")
            self.assertIsNone(store.get_last_user_at())
            store.close()

    def test_records_and_reads_proactive_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PresenceStore(Path(temp_dir) / "presence.db")
            ts = datetime(2026, 8, 19, 14, 30, 0, tzinfo=timezone.utc)
            store.record_proactive(ts)

            self.assertEqual(store.get_last_proactive_at(), ts)
            store.close()


if __name__ == "__main__":
    unittest.main()
