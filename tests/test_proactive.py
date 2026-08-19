"""主动推送循环测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, UTC
from pathlib import Path

from proactive.loop import ProactiveLoop
from proactive.presence import PresenceStore


class ProactiveLoopTest(unittest.IsolatedAsyncioTestCase):
    def _make_loop(
        self, temp_dir: str, *, busy=False, max_ticks=1
    ) -> tuple[ProactiveLoop, PresenceStore]:
        presence = PresenceStore(Path(temp_dir) / "presence.db")
        presence.record_user_message(
            datetime.now(UTC) - timedelta(hours=6)
        )
        loop = ProactiveLoop(
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
            presence.record_proactive(datetime.now(UTC))

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


class PresenceStoreTest(unittest.TestCase):
    def test_records_and_reads_user_message_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PresenceStore(Path(temp_dir) / "presence.db")
            ts = datetime(2026, 8, 19, 14, 30, 0, tzinfo=UTC)
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
            ts = datetime(2026, 8, 19, 14, 30, 0, tzinfo=UTC)
            store.record_proactive(ts)

            self.assertEqual(store.get_last_proactive_at(), ts)
            store.close()


if __name__ == "__main__":
    unittest.main()
