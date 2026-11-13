"""主动引擎入口测试：config 段透传 + engine 模式装配。"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from mind.config import load_config


def _write_config(temp_dir: str, proactive: str = "") -> str:
    path = Path(temp_dir) / "config.toml"
    path.write_text(
        """
[llm]
main = "rt"

[llm.runtimes.rt]
provider = "openai"
model = "gpt-4o"
api_key = "sk-test"
base_url = "https://api.openai.com/v1"

[workspace]
path = "~/.proactivemind/engine-test"
"""
        + proactive,
        encoding="utf-8",
    )
    return str(path)


class ProactiveRawPassthroughTest(unittest.TestCase):
    """[proactive] 段原始字典透传。"""

    def test_missing_section_defaults_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cfg = load_config(_write_config(temp_dir))
            self.assertEqual(cfg.proactive_raw, {})

    def test_section_transparent(self) -> None:
        proactive = """
[proactive]
enabled = true
profile = "quiet"

[proactive.target]
channel = "telegram"
chat_id = "42"
"""
        with TemporaryDirectory() as temp_dir:
            cfg = load_config(_write_config(temp_dir, proactive))
            self.assertTrue(cfg.proactive_raw.get("enabled"))
            self.assertEqual(cfg.proactive_raw.get("profile"), "quiet")
            self.assertEqual(
                cfg.proactive_raw.get("target", {}).get("chat_id"), "42"
            )

    def test_raw_feeds_proactive_loader(self) -> None:
        """透传的原始段可直接进 load_proactive_config。"""
        from initiative.proactive_config_loader import load_proactive_config

        proactive = """
[proactive]
enabled = true
profile = "daily"

[proactive.target]
channel = "web"
chat_id = "u1"
"""
        with TemporaryDirectory() as temp_dir:
            cfg = load_config(_write_config(temp_dir, proactive))
            engine_cfg = load_proactive_config(cfg.proactive_raw)
            self.assertTrue(engine_cfg.enabled)
            self.assertEqual(engine_cfg.default_channel, "web")
            self.assertEqual(engine_cfg.default_chat_id, "u1")


class EngineEntryModeTest(unittest.TestCase):
    """app.py 的 engine 模式调度。"""

    def test_mode_dispatch_registered(self) -> None:
        import app

        src = inspect.getsource(app)
        self.assertIn('mode == "engine"', src)
        self.assertIn("proactive_engine_entry", src)

    def test_entry_function_exists(self) -> None:
        import app

        self.assertTrue(hasattr(app, "proactive_engine_entry"))
        self.assertTrue(
            inspect.iscoroutinefunction(app.proactive_engine_entry)
        )

    def test_docstring_mentions_config_section(self) -> None:
        import app

        doc = app.proactive_engine_entry.__doc__ or ""
        self.assertIn("[proactive]", doc)


class EngineEntryBehaviorTest(unittest.IsolatedAsyncioTestCase):
    """engine 入口的守卫路径（不真正启动循环）。"""

    async def test_exits_when_no_proactive_section(self) -> None:
        import app

        with TemporaryDirectory() as temp_dir:
            cfg_path = _write_config(temp_dir)  # 无 [proactive]
            fake_cfg = load_config(cfg_path)

            with (
                patch.object(app, "load_config", return_value=fake_cfg),
                patch.object(
                    app.sys, "exit", side_effect=SystemExit(1)
                ) as exit_mock,
            ):
                with self.assertRaises(SystemExit):
                    await app.proactive_engine_entry()
                exit_mock.assert_called_once_with(1)

    async def test_exits_when_config_error(self) -> None:
        import app

        proactive = """
[proactive]
enabled = true
profile = "nonexistent_profile"
"""
        with TemporaryDirectory() as temp_dir:
            cfg_path = _write_config(temp_dir, proactive)
            fake_cfg = load_config(cfg_path)

            with (
                patch.object(app, "load_config", return_value=fake_cfg),
                patch.object(
                    app.sys, "exit", side_effect=SystemExit(1)
                ) as exit_mock,
            ):
                with self.assertRaises(SystemExit):
                    await app.proactive_engine_entry()
                exit_mock.assert_called_once_with(1)

    async def test_exits_when_disabled(self) -> None:
        import app

        proactive = """
[proactive]
enabled = false
profile = "daily"

[proactive.target]
channel = "web"
chat_id = "u"
"""
        with TemporaryDirectory() as temp_dir:
            cfg_path = _write_config(temp_dir, proactive)
            fake_cfg = load_config(cfg_path)

            with (
                patch.object(app, "load_config", return_value=fake_cfg),
                patch.object(
                    app.sys, "exit", side_effect=SystemExit(1)
                ) as exit_mock,
            ):
                with self.assertRaises(SystemExit):
                    await app.proactive_engine_entry()
                exit_mock.assert_called_once_with(1)

    async def test_full_path_builds_and_stops(self) -> None:
        """enabled 配置走到 builder + kernel 生命周期（loop 立即停）。"""
        import app

        proactive = """
[proactive]
enabled = true
profile = "daily"

[proactive.target]
channel = "web"
chat_id = "u1"
"""
        with TemporaryDirectory() as temp_dir:
            cfg_path = _write_config(temp_dir, proactive)
            fake_cfg = load_config(cfg_path)

            kernel = MagicMock()
            kernel.start = AsyncMock()
            kernel.stop = AsyncMock()
            loop = MagicMock()
            loop.run = MagicMock(side_effect=KeyboardInterrupt)

            with (
                patch.object(app, "load_config", return_value=fake_cfg),
                patch(
                    "initiative.proactive_builder.build_proactive_engine",
                    return_value=(kernel, loop),
                ) as build_mock,
                patch(
                    "initiative.presence.PresenceStore"
                ) as presence_cls,
                patch(
                    "initiative.data_sources.DataSourceManager"
                ) as ds_cls,
            ):
                presence_cls.return_value = MagicMock()
                ds_cls.return_value = MagicMock()
                with self.assertRaises(KeyboardInterrupt):
                    await app.proactive_engine_entry()

            build_mock.assert_called_once()
            kernel.start.assert_awaited_once()
            kernel.stop.assert_awaited_once()
            loop.stop.assert_called_once()


class ConfigSampleTest(unittest.TestCase):
    """config.sample.toml 补充 [proactive] 示例段。"""

    def test_sample_contains_proactive_section(self) -> None:
        sample = Path("config.sample.toml").read_text(encoding="utf-8")
        self.assertIn("[proactive]", sample)


if __name__ == "__main__":
    unittest.main()
