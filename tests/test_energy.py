"""电量模型测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from initiative.energy import (
    compute_energy,
    compute_urgency,
    d_energy,
    d_recent,
    next_interval,
    next_tick_from_score,
)


class EnergyModelTest(unittest.TestCase):
    def test_energy_decreases_over_time(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)
        e_now = compute_energy(base, base)
        e_30min = compute_energy(base, base + timedelta(minutes=30))
        e_4h = compute_energy(base, base + timedelta(hours=4))
        e_48h = compute_energy(base, base + timedelta(hours=48))

        self.assertGreater(e_now, e_30min)
        self.assertGreater(e_30min, e_4h)
        self.assertGreater(e_4h, e_48h)
        self.assertLess(e_48h, 0.06)

    def test_urgency_increases_over_time(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)
        u_now = compute_urgency(base, base)
        u_4h = compute_urgency(base, base + timedelta(hours=4))
        u_48h = compute_urgency(base, base + timedelta(hours=48))

        self.assertLess(u_now, u_4h)
        self.assertLess(u_4h, u_48h)
        self.assertLessEqual(u_48h, 1.0)

    def test_interval_shorter_when_user_idle_longer(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)

        intervals_short = [
            next_interval(base, base + timedelta(minutes=5), jitter=0)
            for _ in range(10)
        ]
        intervals_long = [
            next_interval(base, base + timedelta(hours=6), jitter=0)
            for _ in range(10)
        ]

        avg_short = sum(intervals_short) / len(intervals_short)
        avg_long = sum(intervals_long) / len(intervals_long)

        self.assertGreater(avg_short, avg_long)

    def test_interval_never_below_60s(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)
        for hours in [0, 1, 6, 24, 48, 100]:
            interval = next_interval(
                base, base + timedelta(hours=hours), jitter=0.3
            )
            self.assertGreaterEqual(interval, 60.0)


class DEnergyTest(unittest.TestCase):
    def test_d_energy_is_one_minus_energy(self) -> None:
        self.assertAlmostEqual(d_energy(0.0), 1.0)
        self.assertAlmostEqual(d_energy(1.0), 0.0)
        self.assertAlmostEqual(d_energy(0.3), 0.7)

    def test_d_energy_clamps_out_of_range(self) -> None:
        self.assertEqual(d_energy(-0.5), 1.0)
        self.assertEqual(d_energy(1.5), 0.0)

    def test_d_energy_higher_for_lower_energy(self) -> None:
        # 久没互动（低 energy）→ 饥渴度高
        self.assertGreater(d_energy(0.1), d_energy(0.9))


class DRecentTest(unittest.TestCase):
    def test_d_recent_zero_messages(self) -> None:
        self.assertEqual(d_recent(0), 0.0)
        self.assertEqual(d_recent(-3), 0.0)

    def test_d_recent_increases_with_msg_count(self) -> None:
        self.assertLess(d_recent(1), d_recent(5))
        self.assertLess(d_recent(5), d_recent(10))

    def test_d_recent_capped_at_one(self) -> None:
        self.assertLessEqual(d_recent(1000), 1.0)

    def test_d_recent_known_values(self) -> None:
        # scale=10 时：0→0.00, 5→log(6)/log(11)≈0.747, 10→log(11)/log(11)=1.0
        self.assertAlmostEqual(d_recent(0), 0.0)
        self.assertAlmostEqual(d_recent(5), 0.747, places=2)
        self.assertAlmostEqual(d_recent(10), 1.0)

    def test_d_recent_invalid_scale_returns_zero(self) -> None:
        self.assertEqual(d_recent(5, scale=0.5), 0.0)
        self.assertEqual(d_recent(5, scale=1.0), 0.0)


class NextTickFromScoreTest(unittest.TestCase):
    def test_high_score_uses_short_tick(self) -> None:
        # score > threshold → 短间隔
        sec = next_tick_from_score(0.5, tick_s1=600, tick_s0=1200,
                                   tick_jitter=0, threshold=0.20)
        self.assertEqual(sec, 600)

    def test_low_score_uses_long_tick(self) -> None:
        # score ≤ threshold → 长间隔
        sec = next_tick_from_score(0.1, tick_s1=600, tick_s0=1200,
                                   tick_jitter=0, threshold=0.20)
        self.assertEqual(sec, 1200)

    def test_threshold_boundary_inclusive_low_side(self) -> None:
        # score == threshold → 长间隔（≤）
        sec = next_tick_from_score(0.20, tick_s1=600, tick_s0=1200,
                                   tick_jitter=0, threshold=0.20)
        self.assertEqual(sec, 1200)

    def test_jitter_zero_returns_base(self) -> None:
        for _ in range(20):
            self.assertEqual(
                next_tick_from_score(0.5, tick_s1=600, tick_s0=1200,
                                     tick_jitter=0.0, threshold=0.20),
                600,
            )

    def test_jitter_within_range(self) -> None:
        for _ in range(50):
            sec = next_tick_from_score(0.5, tick_s1=600, tick_s0=1200,
                                       tick_jitter=0.3, threshold=0.20)
            # tick_s1 * (1 - 0.3) = 420 到 tick_s1 * (1 + 0.3) = 780
            self.assertGreaterEqual(sec, 420)
            self.assertLessEqual(sec, 780)

    def test_result_at_least_one(self) -> None:
        for _ in range(50):
            sec = next_tick_from_score(0.5, tick_s1=1, tick_s0=1,
                                       tick_jitter=0.99, threshold=0.20)
            self.assertGreaterEqual(sec, 1)


if __name__ == "__main__":
    unittest.main()
