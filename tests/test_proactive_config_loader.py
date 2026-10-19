"""主动链路配置加载与校验测试。"""

from __future__ import annotations

import unittest

from initiative.proactive_config import ProactiveConfig, config_field_names
from initiative.proactive_config_loader import (
    ProactiveConfigError,
    load_proactive_config,
)


def _base_dict(**overrides):
    base = {
        "enabled": True,
        "profile": "daily",
        "target": {"channel": "telegram", "chat_id": "12345"},
        "model": "gpt-test",
    }
    base.update(overrides)
    return base


class ProactiveConfigDataclassTest(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ProactiveConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.default_channel, "telegram")
        self.assertEqual(cfg.judge_send_threshold, 0.60)
        self.assertEqual(cfg.tick_interval_s0, 4800)

    def test_field_count(self) -> None:
        # 33 个字段覆盖四组参数
        self.assertEqual(len(config_field_names()), 33)

    def test_replace_returns_new_instance(self) -> None:
        cfg = ProactiveConfig()
        cfg2 = cfg.replace(enabled=True, model="m1")
        self.assertTrue(cfg2.enabled)
        self.assertEqual(cfg2.model, "m1")
        # 原实例不变
        self.assertFalse(cfg.enabled)


class LoadBasicTest(unittest.TestCase):
    def test_minimal_valid(self) -> None:
        cfg = load_proactive_config(_base_dict())
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.default_channel, "telegram")
        self.assertEqual(cfg.default_chat_id, "12345")
        self.assertEqual(cfg.model, "gpt-test")

    def test_daily_preset_defaults(self) -> None:
        cfg = load_proactive_config(_base_dict(profile="daily"))
        # daily 预设的 gate 阈值
        self.assertEqual(cfg.judge_send_threshold, 0.60)

    def test_quiet_preset(self) -> None:
        cfg = load_proactive_config(_base_dict(profile="quiet"))
        # quiet 预设阈值更高
        self.assertEqual(cfg.judge_send_threshold, 0.75)

    def test_strategy_params_merged(self) -> None:
        cfg = load_proactive_config(_base_dict())
        # STRATEGY_PARAMS 内置参数应被合入
        self.assertGreater(cfg.score_weight_energy, 0)
        self.assertTrue(cfg.message_dedupe_enabled)

    def test_disabled_by_default(self) -> None:
        cfg = load_proactive_config({
            "profile": "daily",
            "target": {"channel": "web", "chat_id": "1"},
        })
        self.assertFalse(cfg.enabled)


class RootKeyValidationTest(unittest.TestCase):
    def test_forbidden_root_key_rejected(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(_base_dict(tick_interval_s0=100))
        self.assertIn("非法的根级键", str(ctx.exception))
        self.assertIn("tick_interval_s0", str(ctx.exception))

    def test_non_dict_rejected(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config("not a dict")  # type: ignore[arg-type]

    def test_target_must_be_dict(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(target="oops"))


class ProfileValidationTest(unittest.TestCase):
    def test_profile_required(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config({"enabled": True})
        self.assertIn("必填", str(ctx.exception))

    def test_unknown_profile_rejected(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(_base_dict(profile="nonexistent"))
        self.assertIn("无效的 profile", str(ctx.exception))

    def test_custom_profile_merged(self) -> None:
        p = _base_dict(
            profiles={
                "daily": {"gate": {"judge_send_threshold": 0.90}},
            },
        )
        cfg = load_proactive_config(p)
        # 自定义 profile 覆盖内置 daily
        self.assertEqual(cfg.judge_send_threshold, 0.90)

    def test_profiles_must_be_dict_of_dicts(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(profiles={"bad": "oops"}))


class OverridesValidationTest(unittest.TestCase):
    def test_valid_override_applied(self) -> None:
        p = _base_dict(overrides={"gate": {"judge_send_threshold": 0.80}})
        cfg = load_proactive_config(p)
        self.assertEqual(cfg.judge_send_threshold, 0.80)

    def test_unknown_category_rejected(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(_base_dict(overrides={"bogus": {"x": 1}}))
        self.assertIn("非法类别", str(ctx.exception))

    def test_unknown_key_in_category_rejected(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(
                _base_dict(overrides={"gate": {"bogus_key": 1}})
            )
        self.assertIn("不在白名单", str(ctx.exception))

    def test_override_must_be_dict(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(overrides={"gate": "oops"}))


class RangeValidationTest(unittest.TestCase):
    def test_threshold_out_of_range(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(
                _base_dict(overrides={"gate": {"judge_send_threshold": 1.5}})
            )
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_pmin_greater_than_pmax(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(
                _base_dict(
                    overrides={
                        "anyaction": {
                            "anyaction_probability_min": 0.8,
                            "anyaction_probability_max": 0.2,
                        }
                    }
                )
            )
        self.assertIn("不能大于", str(ctx.exception))

    def test_tick_intervals_must_descend(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(
                _base_dict(
                    overrides={
                        "trigger": {
                            "tick_interval_s0": 100,
                            "tick_interval_s1": 200,
                        }
                    }
                )
            )
        self.assertIn("s0 >= s1", str(ctx.exception))

    def test_tick_interval_s1_at_least_one(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(
                _base_dict(
                    overrides={
                        "trigger": {
                            "tick_interval_s0": 100,
                            "tick_interval_s1": 0,
                        }
                    }
                )
            )
        self.assertIn(">= 1", str(ctx.exception))


class AgentDriftSectionTest(unittest.TestCase):
    def test_agent_section_applied(self) -> None:
        p = _base_dict(agent={"model": "fast-model", "max_steps": 10})
        cfg = load_proactive_config(p)
        self.assertEqual(cfg.agent_tick_model, "fast-model")
        self.assertEqual(cfg.agent_tick_max_steps, 10)

    def test_agent_model_falls_back_to_root(self) -> None:
        cfg = load_proactive_config(_base_dict(agent={}))
        self.assertEqual(cfg.agent_tick_model, "gpt-test")

    def test_agent_forbidden_key(self) -> None:
        with self.assertRaises(ProactiveConfigError) as ctx:
            load_proactive_config(_base_dict(agent={"bogus": 1}))
        self.assertIn("proactive.agent", str(ctx.exception))

    def test_drift_section_applied(self) -> None:
        p = _base_dict(drift={"enabled": True, "max_steps": 5})
        cfg = load_proactive_config(p)
        self.assertTrue(cfg.drift_enabled)
        self.assertEqual(cfg.drift_max_steps, 5)

    def test_drift_forbidden_key(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(drift={"bogus": 1}))

    def test_agent_must_be_dict(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(agent="oops"))

    def test_drift_must_be_dict(self) -> None:
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(_base_dict(drift="oops"))


class UnknownKeysFilteredTest(unittest.TestCase):
    def test_extra_keys_in_preset_ignored(self) -> None:
        # 自定义 profile 带未知键时，不在 ProactiveConfig 里的键被过滤
        p = _base_dict(
            profiles={
                "daily": {"gate": {"judge_send_threshold": 0.7,
                                    "unknown_key": 123}},
            },
        )
        # unknown_key 不在 gate 白名单，应被 _validate_overrides 拒绝
        with self.assertRaises(ProactiveConfigError):
            load_proactive_config(p)


class PresetOverridePrecedenceTest(unittest.TestCase):
    def test_override_beats_custom_profile(self) -> None:
        p = _base_dict(
            profiles={"daily": {"gate": {"judge_send_threshold": 0.90}}},
            overrides={"gate": {"judge_send_threshold": 0.95}},
        )
        cfg = load_proactive_config(p)
        self.assertEqual(cfg.judge_send_threshold, 0.95)


if __name__ == "__main__":
    unittest.main()
