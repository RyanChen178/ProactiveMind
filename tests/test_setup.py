"""Setup 向导测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from bootstrap.setup import (
    SetupError,
    WizardAnswers,
    _ask,
    _ask_yes_no,
    render_config_toml,
    run_setup_from_args,
    run_setup_interactive,
    save_config,
)


class AskTest(unittest.TestCase):
    """单个问题询问逻辑。"""

    def test_returns_default_when_empty(self) -> None:
        result = _ask("Q", "default", lambda _: "")
        self.assertEqual(result, "default")

    def test_returns_user_input_when_provided(self) -> None:
        result = _ask("Q", "default", lambda _: "user_input")
        self.assertEqual(result, "user_input")

    def test_strips_whitespace(self) -> None:
        result = _ask("Q", "default", lambda _: "  hello  ")
        self.assertEqual(result, "hello")

    def test_required_raises_when_empty_and_no_default(self) -> None:
        with self.assertRaises(ValueError):
            _ask("Q", "", lambda _: "", required=True)


class AskYesNoTest(unittest.TestCase):
    """是/否问题。"""

    def test_default_yes_with_empty(self) -> None:
        self.assertTrue(_ask_yes_no("Q", True, lambda _: ""))
        self.assertFalse(_ask_yes_no("Q", False, lambda _: ""))

    def test_explicit_yes(self) -> None:
        self.assertTrue(_ask_yes_no("Q", False, lambda _: "y"))
        self.assertTrue(_ask_yes_no("Q", False, lambda _: "YES"))
        self.assertTrue(_ask_yes_no("Q", False, lambda _: "是"))

    def test_explicit_no(self) -> None:
        self.assertFalse(_ask_yes_no("Q", True, lambda _: "n"))
        self.assertFalse(_ask_yes_no("Q", True, lambda _: "no"))


class RenderConfigTest(unittest.TestCase):
    """TOML 内容渲染。"""

    def test_minimal_config(self) -> None:
        answers = WizardAnswers(main_api_key="sk-test")
        toml = render_config_toml(answers)

        # 必有字段
        self.assertIn("[llm]", toml)
        self.assertIn("[llm.runtimes.main_runtime]", toml)
        self.assertIn('provider = "deepseek"', toml)
        self.assertIn('api_key = "sk-test"', toml)
        self.assertIn("[workspace]", toml)
        self.assertIn("[consolidation]", toml)
        self.assertIn("[extensions]", toml)
        self.assertIn("[prompt]", toml)
        # 不应有 telegram 段（未启用）
        self.assertNotIn("[telegram]", toml)
        # 不应有 fast_runtime 段
        self.assertNotIn("fast_runtime", toml)

    def test_with_fast_runtime(self) -> None:
        answers = WizardAnswers(
            main_api_key="sk-main",
            enable_fast_runtime=True,
            fast_api_key="sk-fast",
        )
        toml = render_config_toml(answers)
        self.assertIn("fast = \"fast_runtime\"", toml)
        self.assertIn("[llm.runtimes.fast_runtime]", toml)
        self.assertIn('api_key = "sk-fast"', toml)

    def test_with_telegram(self) -> None:
        answers = WizardAnswers(
            main_api_key="sk-test",
            enable_telegram=True,
            telegram_token="bot-token-123",
        )
        toml = render_config_toml(answers)
        self.assertIn("[telegram]", toml)
        self.assertIn("enabled = true", toml.split("[telegram]")[1])
        self.assertIn('token = "bot-token-123"', toml)

    def test_extensions_disabled(self) -> None:
        answers = WizardAnswers(main_api_key="x", enable_extensions=False)
        toml = render_config_toml(answers)
        self.assertIn('dir = ""', toml)


class SaveConfigTest(unittest.TestCase):
    """配置文件保存。"""

    def test_save_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            result = save_config("hello = 1\n", target)
            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), "hello = 1\n")

    def test_save_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "subdir" / "config.toml"
            save_config("x = 1\n", target)
            self.assertTrue(target.exists())

    def test_refuses_overwrite_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            target.write_text("existing", encoding="utf-8")
            with self.assertRaises(SetupError):
                save_config("new", target, overwrite=False)

    def test_overwrite_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            target.write_text("existing", encoding="utf-8")
            save_config("new", target, overwrite=True)
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_injected_exists_fn_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            exists_fn = MagicMock(return_value=True)
            with self.assertRaises(SetupError):
                save_config("x", target, file_exists_fn=exists_fn)


class InteractiveSetupTest(unittest.TestCase):
    """交互式模式端到端。"""

    def test_full_interactive_with_minimal_answers(self) -> None:
        # 用脚本化输入驱动向导
        inputs = iter([
            "",           # workspace (use default)
            "",           # main provider (default)
            "",           # main model (default)
            "sk-key",     # main api_key (required)
            "",           # main base_url (default)
            "n",          # enable fast runtime? No
            "n",          # enable telegram? No
            "y",          # enable extensions? Yes
            "",           # extensions dir (default)
            "",           # output path (default)
            "n",          # overwrite? No
        ])
        output: list[str] = []

        answers = run_setup_interactive(
            input_fn=lambda _: next(inputs),
            print_fn=output.append,
        )

        self.assertEqual(answers.main_api_key, "sk-key")
        self.assertFalse(answers.enable_fast_runtime)
        self.assertFalse(answers.enable_telegram)
        self.assertTrue(answers.enable_extensions)
        # 进度信息应已打印
        self.assertTrue(any("首次配置向导" in line for line in output))

    def test_interactive_enables_telegram(self) -> None:
        inputs = iter([
            "",           # workspace
            "",           # provider
            "",           # model
            "sk-key",     # api_key
            "",           # base_url
            "n",          # fast?
            "y",          # telegram?
            "bot-token",  # telegram_token
            "n",          # extensions?
            "config.toml",
            "n",          # overwrite?
        ])
        answers = run_setup_interactive(
            input_fn=lambda _: next(inputs),
            print_fn=lambda _: None,
        )

        self.assertTrue(answers.enable_telegram)
        self.assertEqual(answers.telegram_token, "bot-token")
        self.assertFalse(answers.enable_extensions)


class NonInteractiveSetupTest(unittest.TestCase):
    """非交互模式（CLI 参数直接传入）。"""

    def test_non_interactive_minimal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            result = run_setup_from_args(
                {
                    "non_interactive": True,
                    "api_key": "sk-test",
                    "provider": "openai",
                    "model": "gpt-5",
                    "output": str(target),
                },
                print_fn=lambda _: None,
            )

            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            content = target.read_text(encoding="utf-8")
            self.assertIn('provider = "openai"', content)
            self.assertIn('model = "gpt-5"', content)
            self.assertIn('api_key = "sk-test"', content)

    def test_non_interactive_telegram_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            run_setup_from_args(
                {
                    "non_interactive": True,
                    "api_key": "sk-test",
                    "enable_telegram": True,
                    "telegram_token": "tg-secret",
                    "output": str(target),
                },
                print_fn=lambda _: None,
            )

            content = target.read_text(encoding="utf-8")
            self.assertIn("[telegram]", content)
            self.assertIn('token = "tg-secret"', content)

    def test_non_interactive_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            target.write_text("existing", encoding="utf-8")

            with self.assertRaises(SetupError):
                run_setup_from_args(
                    {
                        "non_interactive": True,
                        "api_key": "sk-test",
                        "output": str(target),
                    },
                    print_fn=lambda _: None,
                )

    def test_non_interactive_overwrite_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.toml"
            target.write_text("existing", encoding="utf-8")

            run_setup_from_args(
                {
                    "non_interactive": True,
                    "api_key": "sk-test",
                    "output": str(target),
                    "overwrite": True,
                },
                print_fn=lambda _: None,
            )

            content = target.read_text(encoding="utf-8")
            self.assertNotEqual(content, "existing")
            self.assertIn('api_key = "sk-test"', content)


if __name__ == "__main__":
    unittest.main()