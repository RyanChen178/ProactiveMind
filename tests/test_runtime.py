"""运行时系统测试。"""

import os
import tempfile
import unittest
from pathlib import Path

from mind.config import load_config, validate_config
from mind.runtime import (
    RuntimeConfig,
    RuntimeRegistry,
    load_runtime_registry,
    _expand_env_var,
)


class TestRuntimeConfig(unittest.TestCase):

    def test_valid_runtime_config(self):
        config = RuntimeConfig(
            runtime_id="test",
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
            context_window=128000,
            max_output_tokens=4096,
        )
        self.assertEqual(config.runtime_id, "test")
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.context_window, 128000)

    def test_invalid_context_window(self):
        with self.assertRaises(ValueError) as ctx:
            RuntimeConfig(
                runtime_id="test",
                provider="deepseek",
                model="deepseek-chat",
                api_key="sk-test",
                base_url="https://api.deepseek.com/v1",
                context_window=0,
            )
        self.assertIn("context_window", str(ctx.exception))

    def test_invalid_max_output_tokens(self):
        with self.assertRaises(ValueError) as ctx:
            RuntimeConfig(
                runtime_id="test",
                provider="deepseek",
                model="deepseek-chat",
                api_key="sk-test",
                base_url="https://api.deepseek.com/v1",
                context_window=1000,
                max_output_tokens=1000,
            )
        self.assertIn("max_output_tokens", str(ctx.exception))

    def test_missing_text_modality(self):
        with self.assertRaises(ValueError) as ctx:
            RuntimeConfig(
                runtime_id="test",
                provider="deepseek",
                model="deepseek-chat",
                api_key="sk-test",
                base_url="https://api.deepseek.com/v1",
                input_modalities=("image",),
            )
        self.assertIn("input_modalities", str(ctx.exception))


class TestRuntimeRegistry(unittest.TestCase):

    def test_valid_registry(self):
        runtime1 = RuntimeConfig(
            runtime_id="main",
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-test",
            base_url="https://api.deepseek.com/v1",
        )
        runtime2 = RuntimeConfig(
            runtime_id="fast",
            provider="qwen",
            model="qwen-flash",
            api_key="sk-test2",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        registry = RuntimeRegistry(
            runtimes={"main": runtime1, "fast": runtime2},
            main="main",
            fast="fast",
        )
        self.assertEqual(registry.get_main().runtime_id, "main")
        self.assertEqual(registry.get_fast().runtime_id, "fast")
        self.assertIsNone(registry.get_vl())

    def test_invalid_main_reference(self):
        with self.assertRaises(ValueError) as ctx:
            RuntimeRegistry(runtimes={}, main="nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_list_names(self):
        runtime1 = RuntimeConfig(
            runtime_id="beta", provider="deepseek", model="deepseek-chat",
            api_key="sk-test", base_url="https://api.deepseek.com/v1",
        )
        runtime2 = RuntimeConfig(
            runtime_id="alpha", provider="qwen", model="qwen-flash",
            api_key="sk-test2", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        registry = RuntimeRegistry(runtimes={"beta": runtime1, "alpha": runtime2}, main="beta")
        self.assertEqual(registry.list_names(), ["alpha", "beta"])

class TestLoadRuntimeRegistry(unittest.TestCase):

    def test_load_multiple_runtimes(self):
        data = {
            "llm": {
                "main": "deepseek_main",
                "fast": "qwen_fast",
                "runtimes": {
                    "deepseek_main": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "api_key": "sk-test",
                        "base_url": "https://api.deepseek.com/v1",
                        "context_window": 128000,
                    },
                    "qwen_fast": {
                        "provider": "qwen",
                        "model": "qwen-flash",
                        "api_key": "sk-test2",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    },
                },
            }
        }
        registry = load_runtime_registry(data)
        self.assertEqual(registry.main, "deepseek_main")
        self.assertEqual(registry.fast, "qwen_fast")
        self.assertEqual(len(registry.runtimes), 2)
        self.assertEqual(registry.get_main().model, "deepseek-v4-flash")

    def test_expand_env_var(self):
        os.environ["TEST_API_KEY"] = "sk-from-env"
        result = _expand_env_var("${TEST_API_KEY}")
        self.assertEqual(result, "sk-from-env")
        del os.environ["TEST_API_KEY"]

    def test_load_with_env_var(self):
        os.environ["DEEPSEEK_KEY"] = "sk-env-key"
        data = {
            "llm": {
                "main": "main",
                "runtimes": {
                    "main": {
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "api_key": "${DEEPSEEK_KEY}",
                        "base_url": "https://api.deepseek.com/v1",
                    },
                },
            }
        }
        registry = load_runtime_registry(data)
        self.assertEqual(registry.get_main().api_key, "sk-env-key")
        del os.environ["DEEPSEEK_KEY"]


class TestConfigIntegration(unittest.TestCase):

    def _write_toml(self, path, sections):
        lines = []
        for section, kvs in sections.items():
            lines.append(f"[{section}]")
            for k, v in kvs.items():
                if isinstance(v, str):
                    lines.append(f"{k} = \"{v}\"")
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        path.write_text(chr(10).join(lines), encoding="utf-8")

    def test_load_config_with_runtimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            self._write_toml(config_path, {
                "llm": {"main": "deepseek_main"},
                "llm.runtimes.deepseek_main": {
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "api_key": "sk-test",
                    "base_url": "https://api.deepseek.com/v1",
                    "context_window": 128000,
                },
                "workspace": {"path": "~/.proactivemind/workspace"},
                "context.compaction": {"keep_recent_tokens": 20000},
            })
            config = load_config(str(config_path))
            self.assertEqual(config.runtimes.main, "deepseek_main")
            self.assertEqual(config.context_compaction.keep_recent_tokens, 20000)
            self.assertEqual(config.llm.model, "deepseek-v4-flash")

    def test_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            self._write_toml(config_path, {
                "llm": {
                    "api_key": "sk-old-format",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "max_tokens": 4096,
                },
                "workspace": {"path": "~/.proactivemind/workspace"},
            })
            config = load_config(str(config_path))
            self.assertEqual(config.llm.model, "deepseek-chat")
            self.assertEqual(config.llm.api_key, "sk-old-format")

    def test_validate_config_with_runtimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            self._write_toml(config_path, {
                "llm": {"main": "main"},
                "llm.runtimes.main": {
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                    "api_key": "sk-test",
                    "base_url": "https://api.deepseek.com/v1",
                },
                "workspace": {"path": "~/.proactivemind/workspace"},
            })
            config = load_config(str(config_path))
            problems = validate_config(config)
            self.assertEqual(len(problems), 0)


if __name__ == "__main__":
    unittest.main()