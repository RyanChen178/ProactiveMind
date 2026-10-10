"""JSON 抽取工具 —— 从 LLM 响应中安全取出 JSON 对象。

设计思想（与 a20b45af proactive_v2/json_utils.py 对齐）：

  - LLM 经常把 JSON 包在 markdown 代码块（```json ... ```）或
    加前后说明文字，本工具先剥壳再匹配
  - 正则找最外层 {...}，保证即便夹了杂文本也能提取
  - 解析后必须是 dict（list/标量都拒绝），类型契约清晰
  - 失败抛 ValueError 而非静默返回 None，调用方必须显式处理

ProactiveMind 命名（initiative/json_extract.py）独立于原仓库。
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_text(text: str) -> str:
    """从文本中抽出最可能的 JSON 字符串。

    处理策略：
      1. 去掉首尾空白
      2. 如果以 ``` 开头（markdown 代码块），剥掉外层 ```
      3. 用正则找最外层 {...}
      4. 找不到时返回原文本
    """
    payload = (text or "").strip()
    if payload.startswith("```"):
        # 去掉首尾的代码块标记
        after_open = payload.split("\n", 1)[-1]
        payload = after_open.rsplit("```", 1)[0].strip()
    match = _JSON_OBJECT_RE.search(payload)
    if match:
        return match.group()
    return payload


def extract_json_object(text: str) -> dict[str, Any]:
    """从文本中抽 JSON 并解析为 dict。失败抛 ValueError。

    非 dict 类型（list/str/number/null）都拒绝。
    """
    data = json.loads(extract_json_text(text))
    if not isinstance(data, dict):
        raise ValueError(f"json payload is not an object: {type(data).__name__}")
    return data
