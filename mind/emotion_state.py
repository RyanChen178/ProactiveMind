"""主动链路 VAD 情绪状态 —— 持久化与衰减。

设计思想（与 akashic-agent a20b45af emotion plugin 对齐）：

  - 三维 VAD：Valence（效价，正负情感）/ Arousal（唤醒度）/ Dominance（支配感）
  - 状态持久化到 SQLite（每个 workspace 一个 emotion.db）
  - 时间衰减：长时间无信号时情绪向中性 (0,0,0) 回落（指数衰减）
  - 反馈驱动增量：每条反馈按类型/置信度映射成 (Δvalence, Δdominance)
  - 完整事件审计：emotion_events 表记录每条变更的 before/after/原因
  - 每 tick 一次 effect 快照：emotion_effects 表记录当时阈值调整依据

ProactiveMind 的命名/路径与参考项目独立，但核心机制（VAD + 衰减 +
反馈更新 + SQLite WAL）一致。
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 衰减半衰期（小时）：距上次更新 N 小时，状态向中性回落的强度系数
DECAY_HALF_LIFE_HOURS = 6.0
# 状态各维度上下限
VALUE_MIN, VALUE_MAX = -1.0, 1.0
# 反馈类型 → (valence 增量, dominance 增量, reason 标签)
FEEDBACK_DELTAS: dict[str, tuple[float, float, str]] = {
    "explicit_quote": (0.03, 0.08, "explicit_quote"),
    "topic_follow_high": (0.02, 0.05, "topic_follow_high"),
    "topic_follow_medium": (0.01, 0.03, "topic_follow_medium"),
    "no_topic_follow": (-0.02, -0.03, "no_topic_follow"),
    "neutral": (0.0, 0.0, "neutral_feedback"),
}

# 反馈 → (阈值下限, 阈值上限) —— 阈值下限应高阈值 delta 调整
_MIN_DELTA_VALUE = -0.05
_MAX_DELTA_VALUE = 0.10


@dataclass(frozen=True)
class EmotionState:
    valence: float
    arousal: float
    dominance: float
    updated_at: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class EmotionEffect:
    """一轮 tick 时由情绪状态计算出的推送阈值调整依据。"""

    tick_id: str
    session_key: str
    valence: float
    arousal: float
    dominance: float
    base_threshold: float
    final_threshold: float
    threshold_delta: float
    tone_label: str
    expected_effect: str
    prompt_section: str


def open_db(path: Path) -> sqlite3.Connection:
    """初始化 SQLite 数据库（WAL 模式），返回连接。

    三个表：emotion_state（单行）、emotion_events（事件审计）、emotion_effects（tick 快照）
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS emotion_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            dominance REAL NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            source_plugin TEXT NOT NULL,
            source_event_id TEXT NOT NULL UNIQUE,
            source_type TEXT NOT NULL,
            session_key TEXT NOT NULL,
            valence_before REAL NOT NULL,
            arousal_before REAL NOT NULL,
            dominance_before REAL NOT NULL,
            valence_delta REAL NOT NULL,
            arousal_delta REAL NOT NULL,
            dominance_delta REAL NOT NULL,
            valence_after REAL NOT NULL,
            arousal_after REAL NOT NULL,
            dominance_after REAL NOT NULL,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotion_effects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tick_id TEXT NOT NULL UNIQUE,
            session_key TEXT NOT NULL,
            valence REAL NOT NULL,
            arousal REAL NOT NULL,
            dominance REAL NOT NULL,
            base_threshold REAL NOT NULL,
            final_threshold REAL NOT NULL,
            threshold_delta REAL NOT NULL,
            tone_label TEXT NOT NULL,
            expected_effect TEXT NOT NULL,
            prompt_section TEXT NOT NULL
        );
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR IGNORE INTO emotion_state(id, valence, arousal, dominance, updated_at)
        VALUES(1, 0.0, 0.0, 0.0, ?)
        """,
        (now,),
    )
    conn.commit()
    return conn


def get_state(conn: sqlite3.Connection) -> EmotionState:
    """读取当前 VAD 状态（首次访问时初始化为中性）。"""
    row = conn.execute(
        "SELECT valence, arousal, dominance, updated_at FROM emotion_state WHERE id = 1"
    ).fetchone()
    return EmotionState(
        valence=float(row["valence"]),
        arousal=float(row["arousal"]),
        dominance=float(row["dominance"]),
        updated_at=str(row["updated_at"]),
    )


def _clamp(value: float, lo: float = VALUE_MIN, hi: float = VALUE_MAX) -> float:
    return max(lo, min(hi, value))


def _decay(state: EmotionState, now: datetime) -> EmotionState:
    """按时间衰减：与上次更新的时间差越大，越靠近中性。"""
    try:
        last = datetime.fromisoformat(state.updated_at)
    except ValueError:
        return EmotionState(0.0, 0.0, 0.0, now.isoformat())
    elapsed_h = max(0.0, (now - last).total_seconds() / 3600.0)
    if elapsed_h <= 0:
        return state
    # 半衰期指数衰减：保留比例 = exp(-ln2 * elapsed / half_life)
    keep = math.exp(-math.log(2) * elapsed_h / DECAY_HALF_LIFE_HOURS)
    return EmotionState(
        valence=state.valence * keep,
        arousal=state.arousal * keep,
        dominance=state.dominance * keep,
        updated_at=state.updated_at,
    )


def classify_feedback_delta(feedback_type: str, confidence: str = "medium") -> tuple[float, float, str]:
    """把反馈类型映射成 (Δvalence, Δdominance, reason 标签)。"""
    key = feedback_type
    if feedback_type == "topic_follow":
        key = f"topic_follow_{confidence}" if confidence in {"gold", "high"} else "topic_follow_medium"
    if key not in FEEDBACK_DELTAS:
        key = "neutral"
    return FEEDBACK_DELTAS[key]


def apply_feedback(
    conn: sqlite3.Connection,
    *,
    source_event_id: str,
    source_plugin: str,
    session_key: str,
    feedback_type: str,
    confidence: str = "medium",
    payload: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> EmotionState:
    """应用一条反馈到情绪状态，写审计事件并返回新状态。

    流程：读取 → 衰减 → 加 delta → 钳值 → 写回 + 记录审计。
    同 source_event_id 重复调用幂等（UNIQUE 约束）。
    """
    now = now or datetime.now(timezone.utc)
    payload = payload or {}
    delta_v, delta_d, reason = classify_feedback_delta(feedback_type, confidence)

    before_raw = get_state(conn)
    before = _decay(before_raw, now)
    after_val = _clamp(before.valence + delta_v)
    after_aro = _clamp(before.arousal)
    after_dom = _clamp(before.dominance + delta_d)
    after = EmotionState(
        valence=after_val,
        arousal=after_aro,
        dominance=after_dom,
        updated_at=now.isoformat(),
    )

    try:
        conn.execute(
            """
            UPDATE emotion_state
            SET valence = ?, arousal = ?, dominance = ?, updated_at = ?
            WHERE id = 1
            """,
            (after.valence, after.arousal, after.dominance, after.updated_at),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO emotion_events
            (source_plugin, source_event_id, source_type, session_key,
             valence_before, arousal_before, dominance_before,
             valence_delta, arousal_delta, dominance_delta,
             valence_after, arousal_after, dominance_after,
             reason, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_plugin, source_event_id, feedback_type, session_key,
                before.valence, before.arousal, before.dominance,
                delta_v, 0.0, delta_d,
                after.valence, after.arousal, after.dominance,
                reason, json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("emotion apply_feedback 失败: %s", exc)
        conn.rollback()
    return after


def _tone_label(state: EmotionState) -> str:
    """把 VAD 数值映射为可读标签。"""
    v = state.valence
    if v > 0.1:
        return "positive"
    if v < -0.1:
        return "negative"
    return "neutral"


def _threshold_delta(state: EmotionState) -> float:
    """情绪阈值调整：积极情绪小幅降门槛（更愿意推），消极情绪小幅升门槛。"""
    # [-1, 1] 映射到 [max_delta_value, min_delta_value]
    return _clamp(-state.valence * 0.05, _MIN_DELTA_VALUE, _MAX_DELTA_VALUE)


def _expected_effect(tone: str) -> str:
    return {
        "positive": "tone_warm",
        "negative": "tone_cautious",
        "neutral": "tone_neutral",
    }.get(tone, "tone_neutral")


def _prompt_section(state: EmotionState, tone: str) -> str:
    if tone == "positive":
        return f"用户当前情绪积极（valence={state.valence:.2f}），可适度主动关怀。"
    if tone == "negative":
        return f"用户当前情绪低落（valence={state.valence:.2f}），保持简洁、不打扰。"
    return f"用户情绪中性（valence={state.valence:.2f}），按需推送。"


def compute_tick_effect(
    conn: sqlite3.Connection,
    *,
    tick_id: str,
    session_key: str,
    base_threshold: float,
    now: datetime | None = None,
) -> EmotionEffect:
    """一轮 tick：读取当前情绪状态 → 计算阈值调整 → 写 effects 快照。"""
    now = now or datetime.now(timezone.utc)
    state = _decay(get_state(conn), now)
    tone = _tone_label(state)
    delta = _threshold_delta(state)
    final_threshold = max(0.0, min(1.0, base_threshold + delta))

    effect = EmotionEffect(
        tick_id=tick_id,
        session_key=session_key,
        valence=state.valence,
        arousal=state.arousal,
        dominance=state.dominance,
        base_threshold=base_threshold,
        final_threshold=final_threshold,
        threshold_delta=delta,
        tone_label=tone,
        expected_effect=_expected_effect(tone),
        prompt_section=_prompt_section(state, tone),
    )

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO emotion_effects
            (tick_id, session_key, valence, arousal, dominance,
             base_threshold, final_threshold, threshold_delta,
             tone_label, expected_effect, prompt_section)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                effect.tick_id, effect.session_key,
                effect.valence, effect.arousal, effect.dominance,
                effect.base_threshold, effect.final_threshold,
                effect.threshold_delta, effect.tone_label,
                effect.expected_effect, effect.prompt_section,
            ),
        )
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("emotion compute_tick_effect 写入失败: %s", exc)
        conn.rollback()
    return effect