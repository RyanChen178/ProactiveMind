"""主动推送评分测试。"""

from __future__ import annotations

import unittest

from initiative.proactive_judge import (
    DEFAULT_SEND_THRESHOLD,
    DEFAULT_WEIGHTS,
    JudgeFeatures,
    JudgeResult,
    ProactiveJudge,
)


class EnergyScoreTest(unittest.TestCase):
    def test_zero_idle_yields_zero(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures(idle_minutes=0)
        result = j.evaluate(f)
        self.assertAlmostEqual(result.per_dimension["energy"], 0.0, places=4)

    def test_long_idle_approaches_one(self) -> None:
        j = ProactiveJudge(idle_scale_minutes=15.0)
        f = JudgeFeatures(idle_minutes=120)
        result = j.evaluate(f)
        # 1 - exp(-120/15) = 1 - exp(-8) ≈ 0.9997
        self.assertGreater(result.per_dimension["energy"], 0.99)

    def test_scale_one_idle_one(self) -> None:
        j = ProactiveJudge(idle_scale_minutes=1.0)
        f = JudgeFeatures(idle_minutes=1)
        result = j.evaluate(f)
        # 1 - exp(-1) ≈ 0.632
        self.assertAlmostEqual(
            result.per_dimension["energy"], 1 - 2.71828 ** -1, places=3,
        )


class RecencyPenaltyTest(unittest.TestCase):
    def test_zero_recent_full_score(self) -> None:
        j = ProactiveJudge(recency_cap=3)
        f = JudgeFeatures(recent_delivery_count=0)
        result = j.evaluate(f)
        self.assertAlmostEqual(result.per_dimension["recency"], 1.0, places=4)

    def test_cap_recent_zero_score(self) -> None:
        j = ProactiveJudge(recency_cap=3)
        f = JudgeFeatures(recent_delivery_count=3)
        result = j.evaluate(f)
        self.assertAlmostEqual(result.per_dimension["recency"], 0.0, places=4)

    def test_half_cap(self) -> None:
        j = ProactiveJudge(recency_cap=4)
        f = JudgeFeatures(recent_delivery_count=2)
        result = j.evaluate(f)
        self.assertAlmostEqual(result.per_dimension["recency"], 0.5, places=4)


class ContextInterestTest(unittest.TestCase):
    def test_context_none_uses_default(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures()  # context_quality=None
        result = j.evaluate(f)
        self.assertEqual(result.per_dimension["context"], 0.5)

    def test_context_out_of_range_clamped(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures(context_quality=2.5)
        result = j.evaluate(f)
        self.assertEqual(result.per_dimension["context"], 1.0)

    def test_interest_negative_clamped(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures(interest_score=-0.5)
        result = j.evaluate(f)
        self.assertEqual(result.per_dimension["interest"], 0.0)


class DecisionThresholdTest(unittest.TestCase):
    def test_low_idle_skips(self) -> None:
        j = ProactiveJudge(send_threshold=0.6)
        f = JudgeFeatures(idle_minutes=0, recent_delivery_count=0,
                         context_quality=0.5, interest_score=0.5)
        result = j.evaluate(f)
        # energy=0, recency=1, context=0.5, interest=0.5
        # 默认权重 0.35*0 + 0.15*1 + 0.25*0.5 + 0.25*0.5 = 0.25
        self.assertLess(result.score, 0.6)
        self.assertEqual(result.decision, "skip")

    def test_long_idle_sends(self) -> None:
        j = ProactiveJudge(send_threshold=0.6)
        f = JudgeFeatures(idle_minutes=120, recent_delivery_count=0,
                         context_quality=0.9, interest_score=0.9)
        result = j.evaluate(f)
        self.assertGreater(result.score, 0.6)
        self.assertEqual(result.decision, "send")

    def test_threshold_boundary(self) -> None:
        # send_threshold=score 应判定 send
        j = ProactiveJudge(send_threshold=0.5)
        f = JudgeFeatures(idle_minutes=30, recent_delivery_count=0,
                         context_quality=0.5, interest_score=0.5)
        result = j.evaluate(f)
        # energy = 1 - exp(-2) ≈ 0.865
        # recency = 1
        # score = 0.35*0.865 + 0.15*1 + 0.25*0.5 + 0.25*0.5
        #      = 0.3028 + 0.15 + 0.125 + 0.125 = 0.7028
        self.assertGreater(result.score, 0.5)
        self.assertEqual(result.decision, "send")


class CustomWeightsTest(unittest.TestCase):
    def test_zero_weights_falls_back_to_uniform(self) -> None:
        j = ProactiveJudge(
            weights={"energy": 0, "recency": 0, "context": 0, "interest": 0},
            idle_scale_minutes=1.0,
            recency_cap=2,
        )
        f = JudgeFeatures(
            idle_minutes=1.0,
            recent_delivery_count=1,
            context_quality=0.5, interest_score=0.5,
        )
        result = j.evaluate(f)
        self.assertAlmostEqual(result.score, 0.533, places=2)

    def test_recency_zero_weight(self) -> None:
        j = ProactiveJudge(weights={"energy": 0.5, "recency": 0,
                                   "context": 0.25, "interest": 0.25})
        f = JudgeFeatures(idle_minutes=0, recent_delivery_count=100,
                         context_quality=0.5, interest_score=0.5)
        # recency=0 但 weight=0 → 不影响 score
        result = j.evaluate(f)
        # energy=0, ctx=0.5, interest=0.5
        # (0.5*0 + 0.25*0.5 + 0.25*0.5) / 1.0 = 0.25
        self.assertAlmostEqual(result.score, 0.25, places=3)


class ShouldSendConvenienceTest(unittest.TestCase):
    def test_returns_bool_and_result(self) -> None:
        j = ProactiveJudge(send_threshold=0.5)
        f = JudgeFeatures(idle_minutes=60, recent_delivery_count=0,
                         context_quality=0.7, interest_score=0.7)
        should, result = j.should_send(f)
        self.assertTrue(should)
        self.assertIsInstance(result, JudgeResult)


class ReasonFieldTest(unittest.TestCase):
    def test_skip_includes_weakest(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures(idle_minutes=0, recent_delivery_count=5,
                         context_quality=0.5, interest_score=0.5)
        result = j.evaluate(f)
        self.assertEqual(result.decision, "skip")
        self.assertIn("低于阈值", result.reason)
        self.assertIn("weakest=", result.reason)

    def test_send_includes_dimensions(self) -> None:
        j = ProactiveJudge()
        f = JudgeFeatures(idle_minutes=120, recent_delivery_count=0,
                         context_quality=0.9, interest_score=0.9)
        result = j.evaluate(f)
        self.assertEqual(result.decision, "send")
        self.assertIn("通过", result.reason)


class DefaultsTest(unittest.TestCase):
    def test_module_constants(self) -> None:
        self.assertIn("energy", DEFAULT_WEIGHTS)
        self.assertIn("context", DEFAULT_WEIGHTS)
        self.assertEqual(DEFAULT_SEND_THRESHOLD, 0.60)


if __name__ == "__main__":
    unittest.main()
