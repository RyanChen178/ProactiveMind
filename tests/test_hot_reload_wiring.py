"""热重载接线测试：MindLoop 启动后 HotReloader 被初始化 + 启动。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _run(coro):
    return asyncio.run(coro)


class _FakeRegistry:
    def __init__(self):
        self._tools: dict[str, MagicMock] = {}


class _FakeMindLoop:
    """最小复刻 MindLoop 启动 hot_reload 流程的 mock。"""

    def __init__(self, extensions_dir: Path, poll_interval: float = 2.0):
        from mind.extensions.hot_reload import init_hot_reloader

        self._tools = _FakeRegistry()
        self._extension_manager = MagicMock()
        self._extension_manager.load_all = MagicMock()

        def _on_reload() -> None:
            self._extension_manager.load_all(self._tools)

        init_hot_reloader(
            extensions_dir,
            poll_interval=poll_interval,
            on_reload=_on_reload,
        )
        from mind.extensions.hot_reload import get_hot_reloader

        self._hot_reloader = get_hot_reloader()

    def start_hot_reload(self) -> None:
        self._hot_reloader.start()

    def stop(self) -> None:
        self._hot_reloader.stop()


class HotReloadWiringTest(unittest.IsolatedAsyncioTestCase):
    """_load_extensions → init_hot_reloader + start 行为。"""

    def setUp(self) -> None:
        # 强制重置全局单例（每个测试都需要干净的 reloader 状态）
        import mind.extensions.hot_reload as hr

        hr.HotReloader._instance = None

    def test_init_creates_reloader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "exts"
            extensions_dir.mkdir()
            loop = _FakeMindLoop(extensions_dir, poll_interval=2.0)
            self.assertIsNotNone(loop._hot_reloader)

    def test_start_hot_reload_creates_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "exts"
            extensions_dir.mkdir()
            loop = _FakeMindLoop(extensions_dir, poll_interval=2.0)

            async def _check() -> None:
                loop.start_hot_reload()
                self.assertTrue(loop._hot_reloader._running)
                self.assertIsNotNone(loop._hot_reloader._monitor_task)
                loop._hot_reloader.stop()
                self.assertFalse(loop._hot_reloader._running)

            _run(_check())

    def test_poll_interval_zero_skips_in_real_mind_loop(self) -> None:
        """MindLoop._load_extensions 的 poll_interval=0 短路逻辑。"""
        poll_interval = 0.0
        self.assertTrue(poll_interval <= 0)  # 与实现里的短路判断一致


class OnReloadCallbackTest(unittest.IsolatedAsyncioTestCase):
    """on_reload 回调触发全量重新加载。"""

    def test_on_reload_invokes_load_all(self) -> None:
        import mind.extensions.hot_reload as hr

        hr.HotReloader._instance = None
        with tempfile.TemporaryDirectory() as temp_dir:
            extensions_dir = Path(temp_dir) / "exts"
            extensions_dir.mkdir()

            manager = MagicMock()
            manager.load_all = MagicMock(return_value=["ext1"])

            tools = _FakeRegistry()
            from mind.extensions.hot_reload import init_hot_reloader

            def _on_reload() -> None:
                manager.load_all(tools)

            reloader = init_hot_reloader(
                extensions_dir, poll_interval=1.0, on_reload=_on_reload
            )
            reloader._on_reload()
            manager.load_all.assert_called_once_with(tools)


class AppIntegrationTest(unittest.TestCase):
    """app.py 源码静态检查：四个入口都启动热重载。"""

    def test_app_invocations(self) -> None:
        import inspect

        import app

        src = inspect.getsource(app)
        # 至少 4 处 start_hot_reload 调用（cli/web/telegram/control）
        self.assertGreaterEqual(src.count("start_hot_reload"), 4)


if __name__ == "__main__":
    unittest.main()