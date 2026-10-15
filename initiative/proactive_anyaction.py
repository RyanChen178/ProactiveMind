"""主动推送 AnyAction 频控 —— 按日配额 + 最小间隔 + idle 概率门。

设计思想（与 a20b45af proactive_v2/anyaction.py 对齐）：

  - QuotaStore：本地 JSON 持久化每日动作计数
    - 按本地时区 + cfg.anyaction_reset_hour_local 划分窗口
    - rollover：跨日自动重置 used=0
  - AnyActionGate.should_act()：
    1. 硬规则：日配额耗尽（remaining=0）→ 阻断
    2. 硬规则：距上次动作 < min_interval → 阻断
    3. 软规则：概率门
       p = p_min + (p_max - p_min) * (1 - exp(-idle_min / idle_scale))
       idle 越长 p 越大，越接近 p_max
    4. 抽样 rng.random() < p 决定是否通过

依赖最小化：cfg 用 duck typing（attr 可取）；rng 接受 random.Random 或 None。

ProactiveMind 命名（initiative/proactive_anyaction.py）独立于原仓库。
不依赖 core.common.timekit / infra.persistence.json_store，用纯 stdlib。
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_QUOTA_FILE = "anyaction_quota.json"


@dataclass
class QuotaSnapshot:
    """配额当前状态快照。"""

    window_key: str
    next_reset_at: datetime
    used: int
    last_action_at: datetime | None


def _safe_zone(name):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _parse_iso(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None


class QuotaStore:
    """本地 JSON 持久化每日动作计数。"""

    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self):
        if not self.path.exists():
            return {
                "version": 1,
                "window_key": "",
                "next_reset_at": "",
                "used": 0,
                "last_action_at": "",
            }
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[anyaction.quota] 读取失败: %s", exc)
            return {
                "version": 1, "window_key": "", "next_reset_at": "",
                "used": 0, "last_action_at": "",
            }
        return {
            "version": int(raw.get("version", 1)),
            "window_key": str(raw.get("window_key", "")),
            "next_reset_at": str(raw.get("next_reset_at", "")),
            "used": int(raw.get("used", 0)),
            "last_action_at": str(raw.get("last_action_at", "")),
        }

    def _save(self):
        try:
            self.path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("[anyaction.quota] 写入失败: %s", exc)

    def _window_meta(self, now_utc, reset_hour, tz):
        local_now = now_utc.astimezone(tz)
        reset_today = local_now.replace(
            hour=reset_hour, minute=0, second=0, microsecond=0
        )
        if local_now >= reset_today:
            start = reset_today
            next_reset = reset_today + timedelta(days=1)
        else:
            start = reset_today - timedelta(days=1)
            next_reset = reset_today
        key = f"{start.date().isoformat()}@{reset_hour:02d}@{getattr(tz, 'key', str(tz))}"
        return key, next_reset

    def _rollover_if_needed(self, window_key, next_reset_at):
        if self._state.get("window_key") == window_key:
            return
        self._state = {
            "version": 1,
            "window_key": window_key,
            "next_reset_at": next_reset_at.isoformat(),
            "used": 0,
            "last_action_at": self._state.get("last_action_at", ""),
        }
        self._save()

    def snapshot(self, *, now_utc, reset_hour, timezone_name):
        tz = _safe_zone(timezone_name)
        window_key, next_reset_local = self._window_meta(now_utc, reset_hour, tz)
        next_reset_utc = next_reset_local.astimezone(timezone.utc)
        self._rollover_if_needed(window_key=window_key, next_reset_at=next_reset_utc)
        return QuotaSnapshot(
            window_key=self._state["window_key"],
            next_reset_at=_parse_iso(self._state["next_reset_at"]) or now_utc,
            used=int(self._state["used"]),
            last_action_at=_parse_iso(self._state.get("last_action_at", "")),
        )

    def record_action(self, *, now_utc, reset_hour, timezone_name):
        snap = self.snapshot(
            now_utc=now_utc, reset_hour=reset_hour, timezone_name=timezone_name
        )
        self._state["window_key"] = snap.window_key
        self._state["next_reset_at"] = snap.next_reset_at.isoformat()
        self._state["used"] = int(self._state.get("used", 0)) + 1
        self._state["last_action_at"] = now_utc.isoformat()
        self._save()


class AnyActionGate:
    """后台 AnyAction 通用层：硬规则 + 概率门。"""

    def __init__(self, *, cfg, quota_store, rng=None):
        self._cfg = cfg
        self._quota = quota_store
        self._rng = rng if rng is not None else random.Random()

    def should_act(self, *, now_utc, last_user_at):
        snap = self._quota.snapshot(
            now_utc=now_utc,
            reset_hour=int(getattr(self._cfg, "anyaction_reset_hour_local", 12) or 12),
            timezone_name=str(
                getattr(self._cfg, "anyaction_timezone", "UTC") or "UTC"
            ),
        )
        max_actions = int(
            getattr(self._cfg, "anyaction_daily_max_actions", 0) or 0
        )
        remaining = max(0, max_actions - snap.used)

        if remaining <= 0:
            return False, {
                "reason": "quota_exhausted",
                "used_today": snap.used,
                "remaining_today": remaining,
            }

        min_interval = float(
            getattr(self._cfg, "anyaction_min_interval_seconds", 0) or 0
        )
        if snap.last_action_at is not None:
            since_last = (now_utc - snap.last_action_at).total_seconds()
            if since_last < min_interval:
                return False, {
                    "reason": "min_interval",
                    "used_today": snap.used,
                    "remaining_today": remaining,
                    "seconds_since_last_action": max(0.0, since_last),
                }

        idle_scale = float(
            getattr(self._cfg, "anyaction_idle_scale_minutes", 15.0) or 15.0
        )
        if last_user_at is not None:
            idle_min = max(0.0, (now_utc - last_user_at).total_seconds() / 60.0)
        else:
            idle_min = idle_scale * 2.0
        idle_factor = 1.0 - math.exp(-idle_min / max(idle_scale, 1.0))
        p_min = float(
            getattr(self._cfg, "anyaction_probability_min", 0.05) or 0.0
        )
        p_max = float(
            getattr(self._cfg, "anyaction_probability_max", 0.30) or 0.0
        )
        p = p_min + (p_max - p_min) * idle_factor
        p = max(0.0, min(1.0, p))
        draw = self._rng.random()
        return draw < p, {
            "reason": "probability",
            "used_today": snap.used,
            "remaining_today": remaining,
            "idle_minutes": idle_min,
            "p_act": p,
            "draw": draw,
        }

    def record_action(self, *, now_utc):
        self._quota.record_action(
            now_utc=now_utc,
            reset_hour=int(getattr(self._cfg, "anyaction_reset_hour_local", 12) or 12),
            timezone_name=str(
                getattr(self._cfg, "anyaction_timezone", "UTC") or "UTC"
            ),
        )
