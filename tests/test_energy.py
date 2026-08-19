"""电量模型测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from proactive.energy import compute_energy, compute_urgency, next_interval


class EnergyModelTest(unittest.TestCase):
    def test_energy_decreases_over_time(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)
        e_now = compute_energy(base, base)
        e_30min = compute_energy(base, base + timedelta(minutes=30))
        e_4h = compute_energy(base, base + timedelta(hours=4))
        e_48h = compute_energy(base, base + timedelta(hours=48))

        # 能量随时间递减
        self.assertGreater(e_now, e_30min)
        self.assertGreater(e_30min, e_4h)
        self.assertGreater(e_4h, e_48h)
        # 48 小时后能量接近 0
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

        # 刚交互完 → 间隔更长（不烦用户）；很久没动 → 间隔更短
        self.assertGreater(avg_short, avg_long)

    def test_interval_never_below_60s(self) -> None:
        base = datetime(2026, 8, 19, 12, 0, 0)
        for hours in [0, 1, 6, 24, 48, 100]:
            interval = next_interval(
                base, base + timedelta(hours=hours), jitter=0.3
            )
            self.assertGreaterEqual(interval, 60.0)


if __name__ == "__main__":
    unittest.main()
