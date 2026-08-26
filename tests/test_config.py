"""配置加载测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mind.config import load_config


class ConfigTest(unittest.TestCase):
    def test_reads_consolidation_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                "[llm]\napi_key = 'test-key'\n\n"
                "[consolidation]\nenabled = false\n",
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertFalse(config.consolidation.enabled)

    def test_rejects_non_boolean_consolidation_switch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                "[llm]\napi_key = 'test-key'\n\n"
                "[consolidation]\nenabled = 'yes'\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "必须是布尔值"):
                load_config(str(config_path))


if __name__ == "__main__":
    unittest.main()
