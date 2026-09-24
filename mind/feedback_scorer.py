"""主动推送反馈评分 —— 引用归一化与分类打标。

设计思想（与参考项目 proactive_feedback/scorer 对齐）：

  1. 文本清洗：合并空白、截断到 max_chars
  2. 引用归一化：去掉 Markdown 标记 / 控制字符 / 统一大小写
     用于文本相似度比对的引用匹配
  3. 解析主动推送消息：识别系统注入的标记"被回复消息 X：【你当前新消息】"
     拆分出 referenced_text 与 current_text
  4. 主动消息识别：通过消息 metadata 中的 proactive 字段
  5. 反馈分类：基于引用文本相似度判定 feedback_type
     - explicit_quote：用户原样引用了推送消息
     - topic_follow：用户继续了推送主题但换措辞
     - no_topic_follow：用户切到了别的话题
     - neutral：无法判定
  6. 置信度：high / medium / low，按匹配相似度分级

ProactiveMind 命名与参考项目独立（mind/feedback_scorer.py），
路径与算法同构。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

# 用户消息中由系统注入的标记，用于拆分"被回复内容"与"当前内容"
QUOTE_MARKER = "【你当前新消息】"
QUOTE_PREFIX = "被回复消息"

# 主动推送消息 metadata 字段
_PROACTIVE_KEYS = ("proactive", "is_proactive")


@dataclass(frozen=True)
class QuoteParts:
    """从主动推送消息中拆分出的引用片段。"""

    quoted_text: str | None
    current_text: str


@dataclass(frozen=True)
class FeedbackScore:
    """一条主动推送消息获得的反馈评分结果。"""

    feedback_type: str  # explicit_quote / topic_follow / no_topic_follow / neutral
    confidence: str  # high / medium / low
    pa_score: float  # 推送内容与用户响应的字面相似度 [0, 1]
    pua_score: float  # 用户主动跳转度 [0, 1]（值越大表示越偏离）
    matched_by: str  # exact / token / topic / none
    candidate_count: int
    reason: str
    lag_seconds: int | None


# 分类阈值（参考项目同款语义）
EXACT_MATCH_THRESHOLD = 0.85  # pa_score ≥ 此值视为 explicit_quote
TOKEN_OVERLAP_HIGH = 0.5  # token 重叠率 ≥ 此值视为 topic_follow
TOKEN_OVERLAP_MEDIUM = 0.25  # token 重叠率中等置信度
SPLIT_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def clean_text(text: str, max_chars: int = 1200) -> str:
    """合并空白、去除首尾空白、可选截断。"""
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed[:max_chars]


def normalize_quote_text(text: str, max_chars: int = 1200) -> str:
    """规范化引用文本：剥离被回复消息包装 + 去 Markdown 标记 + 小写化 + 去标点。"""
    parsed = parse_quote_parts(text)
    base = parsed.quoted_text if parsed.quoted_text else parsed.current_text or text
    cleaned = clean_text(base, max_chars=max_chars).lower()
    cleaned = re.sub(r"[*_`#>\[\]()]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """简单分词：中文字符按单字，英文/数字按词，过滤长度 1 的英文。"""
    if not text:
        return []
    lowered = text.lower()
    # 英文/数字单词
    tokens: list[str] = re.findall(r"[a-z0-9]+", lowered)
    # 汉字逐字
    tokens.extend(ch for ch in lowered if "一" <= ch <= "鿿")
    return tokens


def parse_quote_parts(content: str) -> QuoteParts:
    """拆分系统注入的"被回复消息 X：【你当前新消息】 ..."格式。"""
    if not content:
        return QuoteParts(quoted_text=None, current_text="")
    if QUOTE_MARKER not in content:
        return QuoteParts(quoted_text=None, current_text=content.strip())

    before, after = content.split(QUOTE_MARKER, 1)
    quoted = before
    if QUOTE_PREFIX in quoted:
        quoted = quoted.split(QUOTE_PREFIX, 1)[1]
    if "：" in quoted:
        quoted = quoted.split("：", 1)[1]
    quoted_text = clean_text(quoted, max_chars=300) or None
    return QuoteParts(quoted_text=quoted_text, current_text=after.strip())


def is_proactive_message(extra: dict | str | None) -> bool:
    """判定一条消息是否由主动推送产生。"""
    if not extra:
        return False
    if isinstance(extra, dict):
        for key in _PROACTIVE_KEYS:
            if extra.get(key):
                return True
        return False
    # 字符串格式容错
    lowered = extra.lower()
    return ("proactive" in lowered) and ("true" in lowered)


def _char_overlap(a: str, b: str) -> float:
    """两字符串的字面相似度（按较短字符串归一化）。

    对短中文串采用 token 重叠率（中文单字 token 在长串里的命中比例），
    比纯 LCS 更稳健——不受字符匹配噪音影响。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if not short or not long:
        return 0.0
    long_chars = set(long)
    matched = sum(1 for ch in short if ch in long_chars)
    return min(1.0, matched / len(short))


def _token_overlap(a: str, b: str) -> float:
    """两字符串 token 重叠率（intersection / min(|A|, |B|)）。"""
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    denom = min(len(ta), len(tb))
    if denom <= 0:
        return 0.0
    return len(intersection) / denom


def _compute_lag_seconds(proactive_ts: str | None, user_ts: str | None) -> int | None:
    """计算用户响应与主动推送的时间间隔（秒）。"""
    if not proactive_ts or not user_ts:
        return None
    try:
        from datetime import datetime

        p = datetime.fromisoformat(proactive_ts)
        u = datetime.fromisoformat(user_ts)
        return max(0, int((u - p).total_seconds()))
    except (ValueError, TypeError):
        return None


def score_feedback(
    proactive_content: str,
    user_content: str,
    candidates: Sequence[str] | None = None,
    proactive_ts: str | None = None,
    user_ts: str | None = None,
) -> FeedbackScore:
    """对一对主动推送消息 / 用户响应打分。

    candidates: 同时刻其它候选推送，用于精确匹配（防止内容相似导致的误判）

    评分核心：
      pa_score：推送内容与用户响应的 token 重叠率（按 token 集合 Jaccard）
      matched_by：先查候选集精确命中（防止推送内容相似导致的误归因），
                  再用主文本 token 重叠做分级（exact / token / topic / none）
    """
    candidates = list(candidates or [])
    quoted = parse_quote_parts(user_content)
    user_text = quoted.current_text or user_content

    normalized_proactive = normalize_quote_text(proactive_content)
    normalized_user = normalize_quote_text(user_text)

    pa_score = _token_overlap(normalized_proactive, normalized_user)

    # 在候选集中寻找与用户响应的字面匹配
    matched_by = "none"
    best_match_score = 0.0
    for cand in candidates:
        n = normalize_quote_text(cand)
        score = _char_overlap(n, normalized_user)
        if score > best_match_score:
            best_match_score = score
            if score >= EXACT_MATCH_THRESHOLD:
                matched_by = "exact"

    if matched_by == "none" and pa_score >= EXACT_MATCH_THRESHOLD:
        matched_by = "exact"
    elif matched_by == "none":
        to = _token_overlap(normalized_proactive, normalized_user)
        if to >= TOKEN_OVERLAP_HIGH:
            matched_by = "token"
        elif to >= TOKEN_OVERLAP_MEDIUM:
            matched_by = "topic"

    if matched_by == "exact":
        feedback_type = "explicit_quote"
        confidence = "high"
        reason = "用户引用了原推送内容"
    elif matched_by == "token":
        feedback_type = "topic_follow"
        confidence = "high"
        reason = "用户响应与推送 token 重叠较高"
    elif matched_by == "topic":
        feedback_type = "topic_follow"
        confidence = "medium"
        reason = "用户响应与推送 token 重叠中等"
    else:
        feedback_type = "no_topic_follow"
        confidence = "low"
        reason = "用户响应与推送无明显重叠"

    pua_score = max(0.0, min(1.0, 1.0 - pa_score))
    return FeedbackScore(
        feedback_type=feedback_type,
        confidence=confidence,
        pa_score=round(pa_score, 4),
        pua_score=round(pua_score, 4),
        matched_by=matched_by,
        candidate_count=len(candidates),
        reason=reason,
        lag_seconds=_compute_lag_seconds(proactive_ts, user_ts),
    )


def feedback_score_from_event(event: dict) -> FeedbackScore:
    """从 MindLoop 事件数据 dict 中提取字段并评分（适配 EventHub 事件）。"""
    return score_feedback(
        proactive_content=event.get("proactive_content", ""),
        user_content=event.get("user_content", ""),
        candidates=event.get("candidates") or [],
        proactive_ts=event.get("proactive_ts"),
        user_ts=event.get("user_ts"),
    )