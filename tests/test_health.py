"""健康检查测试。"""

from __future__ import annotations

import unittest

from mind.health import HealthChecker, create_health_checker


class HealthCheckerTest(unittest.TestCase):
    def test_returns_healthy_when_no_checks(self) -> None:
        hc = HealthChecker()
        report = hc.check()
        self.assertEqual(report.status, "healthy")
        self.assertEqual(len(report.components), 0)

    def test_returns_healthy_when_all_pass(self) -> None:
        hc = HealthChecker()
        hc.register("db", lambda: (True, "ok"))
        hc.register("cache", lambda: (True, "ok"))
        report = hc.check()
        self.assertEqual(report.status, "healthy")
        self.assertTrue(report.all_healthy)

    def test_returns_degraded_when_partial_failure(self) -> None:
        hc = HealthChecker()
        hc.register("db", lambda: (True, "ok"))
        hc.register("cache", lambda: (False, "连接超时"))
        report = hc.check()
        self.assertEqual(report.status, "degraded")

    def test_returns_unhealthy_when_all_fail(self) -> None:
        hc = HealthChecker()
        hc.register("db", lambda: (False, "down"))
        hc.register("cache", lambda: (False, "down"))
        report = hc.check()
        self.assertEqual(report.status, "unhealthy")

    def test_catches_exception_in_check_fn(self) -> None:
        hc = HealthChecker()

        def bad_check():
            raise RuntimeError("boom")

        hc.register("bad", bad_check)
        report = hc.check()
        self.assertEqual(report.status, "unhealthy")
        self.assertEqual(len(report.components), 1)
        self.assertFalse(report.components[0].healthy)
        self.assertIn("boom", report.components[0].detail)

    def test_handles_non_boolean_return(self) -> None:
        hc = HealthChecker()
        hc.register("weird", lambda: ("yes", "not boolean"))
        report = hc.check()
        self.assertEqual(report.status, "unhealthy")
        self.assertFalse(report.components[0].healthy)

    def test_to_dict_includes_all_fields(self) -> None:
        hc = HealthChecker()
        hc.register("db", lambda: (True, "ok"))
        report = hc.check()
        d = hc.to_dict(report)
        self.assertEqual(d["status"], "healthy")
        self.assertIn("timestamp", d)
        self.assertEqual(len(d["components"]), 1)
        self.assertEqual(d["components"][0]["name"], "db")
        self.assertTrue(d["components"][0]["healthy"])
        self.assertIn("latency_ms", d["components"][0])

    def test_create_health_checker_with_stores(self) -> None:
        class FakeMemory:
            def read_all(self):
                return ""

        class FakeSessionStore:
            def get_or_create_active_session(self):
                return "test"

        class FakePresence:
            def get_last_user_at(self):
                return None

        hc = create_health_checker(
            memory_store=FakeMemory(),
            session_store=FakeSessionStore(),
            presence_store=FakePresence(),
        )
        report = hc.check()
        self.assertEqual(report.status, "healthy")
        names = [c.name for c in report.components]
        self.assertEqual(names, ["memory", "sessions", "presence"])

    def test_create_health_checker_handles_store_errors(self) -> None:
        class BrokenMemory:
            def read_all(self):
                raise RuntimeError("文件丢失")

        hc = create_health_checker(memory_store=BrokenMemory())
        report = hc.check()
        self.assertEqual(report.status, "unhealthy")
        self.assertFalse(report.components[0].healthy)
        self.assertIn("文件丢失", report.components[0].detail)


class ConfigValidationTest(unittest.TestCase):
    def test_valid_config_has_no_problems(self) -> None:
        from mind.config import Config, PromptConfig, ConsolidationConfig, validate_config
        from mind.runtime import RuntimeConfig, RuntimeRegistry
        from pathlib import Path

        runtime = RuntimeConfig(
            runtime_id="test",
            provider="deepseek",
            model="test-model",
            api_key="test",
            base_url="https://api.test.com/v1",
        )
        registry = RuntimeRegistry(runtimes={"test": runtime}, main="test")
        config = Config(
            runtimes=registry,
            workspace=Path("/tmp"),
            max_history_tokens=6000,
            prompt=PromptConfig(),
            consolidation=ConsolidationConfig(),
        )
        problems = validate_config(config)
        self.assertEqual(problems, [])

    def test_empty_api_key_flagged(self) -> None:
        from mind.config import Config, validate_config
        from mind.runtime import RuntimeConfig, RuntimeRegistry
        from pathlib import Path

        runtime = RuntimeConfig(
            runtime_id="test",
            provider="deepseek",
            model="test",
            api_key="",
            base_url="https://api.test.com/v1",
        )
        registry = RuntimeRegistry(runtimes={"test": runtime}, main="test")
        config = Config(runtimes=registry, workspace=Path("/tmp"))
        problems = validate_config(config)
        self.assertTrue(any("api_key" in p for p in problems))

    def test_small_max_history_tokens_flagged(self) -> None:
        from mind.config import Config, validate_config
        from mind.runtime import RuntimeConfig, RuntimeRegistry
        from pathlib import Path

        runtime = RuntimeConfig(
            runtime_id="test",
            provider="deepseek",
            model="m",
            api_key="k",
            base_url="url",
        )
        registry = RuntimeRegistry(runtimes={"test": runtime}, main="test")
        config = Config(
            runtimes=registry,
            workspace=Path("/tmp"),
            max_history_tokens=50,
        )
        problems = validate_config(config)
        self.assertTrue(any("max_history_tokens" in p for p in problems))

    def test_nonexistent_extensions_dir_flagged(self) -> None:
        from mind.config import Config, validate_config
        from mind.runtime import RuntimeConfig, RuntimeRegistry
        from pathlib import Path

        runtime = RuntimeConfig(
            runtime_id="test",
            provider="deepseek",
            model="m",
            api_key="k",
            base_url="url",
        )
        registry = RuntimeRegistry(runtimes={"test": runtime}, main="test")
        config = Config(
            runtimes=registry,
            workspace=Path("/tmp"),
            extensions_dir=Path("/nonexistent/ext"),
        )
        problems = validate_config(config)
        self.assertTrue(any("extensions_dir" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
