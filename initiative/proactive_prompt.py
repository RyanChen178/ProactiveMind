"""主动链路提示词组装 —— 决策指令 + 运行时上下文 frame。

设计思想（与 a20b45af proactive_v2/modules_prompt.py 对齐）：

  - build_system_prompt(plugin_sections)：
      静态决策指令（角色定位 / 数据优先级 / 分类任务 / 工具职责 /
      规则优先级 / 信息源边界 / 决策路径 / 发送规范），
      中间可插入插件上报的状态区块
  - build_runtime_context_message(tick_state, data)：
      动态上下文 frame，逐段拼装（空段跳过）：
        tick 状态（fallback 开关 + 三路计数）
        自我模型 / 长期记忆 / 近期上下文（memory profile 注入，
        任一段读取异常降级为空，不阻断）
        alerts / content / context 三个数据渲染块
        workspace 主动上下文（函数注入，截断防溢出）
  - 三个渲染器：
      alerts 编号列表（高优置顶说明）
      content 编号列表 + 是否已含正文标注（正文按需工具获取）
      context JSON 序列化 + 截断上限
  - ProactivePromptModule：phase 壳（slot=prompt /
    phase=proactive.prompt），把组装结果写进 frame.slots

指令文本为独立撰写（机制与结构等价，措辞原创）。

ProactiveMind 命名（initiative/proactive_prompt.py）独立于原仓库。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from initiative.proactive_frame import ProactiveFrame

logger = logging.getLogger(__name__)

SLOT_PROMPT_SYSTEM = "prompt:system"
SLOT_PROMPT_CONTEXT = "prompt:context"

WORKSPACE_CONTEXT_LIMIT = 3000
CONTEXT_JSON_LIMIT = 900


@dataclass
class PromptData:
    """一轮 tick 的三路数据快照（gateway 预取结果）。"""

    alerts: list[dict[str, Any]] = field(default_factory=list)
    content_meta: list[dict[str, Any]] = field(default_factory=list)
    content_store: dict[str, str] = field(default_factory=dict)
    context: list[dict[str, Any]] = field(default_factory=list)


DECISION_DIRECTIVE = """你是主动推送决策引擎，任务是判断此刻是否该给用户发消息、发什么。

数据已在后续上下文中预取完毕：Alert（系统级高优通知）、Content（内容流候选）、
Context（背景参考）。三类数据的优先级固定为 Alert > Content > Context，
其中 Context 仅在本轮明确标注允许兜底时才可使用。

主任务是对候选做分类，不是开拓新话题：
- 逐条独立评估 Content，不得把多条不同主题打包成一次判断，
  也不得因前几条不感兴趣就跳过剩余条目。
- 分类工具只有标记感兴趣与标记不感兴趣两种；每条必须落一个结论，
  附一句简短理由（未验证的事实要明确写"未验证"）。

工具职责边界：
- 用户工作区上下文是本轮筛选规则（黑白名单 / 必须先验证的步骤），
  它是规则不是内容源；与常识冲突时以它为准。
- 记忆检索只用于判断用户兴趣与雷点，不能替代规则校验。
- 正文补全与网页抓取用于核实候选细节；抓取失败不代表不感兴趣，
  应回退到以来源名做记忆检索。
- 最近对话只用于最后判断此刻是否适合打扰。
- 消息推送暂存草稿；结束本轮（回复或跳过）才是终止动作。

信息源纪律：
- 最终消息里的事实只能来自本轮提供的三路数据；
  长期记忆、工作区规则、最近对话只用于过滤与排序，
  严禁凭训练知识脑补比分、排名、日期等可验证细节。
- 三路数据均为空时只有两条路：默认静默结束；
  或在最近对话确有自然延伸的未完成话题时轻松接一句
  （此时不得引用任何外部事实）。不允许检索、抓取或捏造条目。

发送规范：
- 语气自然如朋友分享；来源可靠时在对应段落后自然附原始链接，
  不集中堆到末尾，不杜撰链接。
- 静默结束是正当选择，没有实质内容时不要强行发。"""


class ProactivePromptBuilder:
    """主动推送提示词组装器。"""

    def __init__(
        self,
        *,
        memory: Any | None = None,
        workspace_context_fn: Callable[[], str] | None = None,
    ) -> None:
        self._memory = memory
        self._workspace_context_fn = workspace_context_fn

    # ------------------------------------------------------------------
    # 静态指令
    # ------------------------------------------------------------------

    def build_system_prompt(
        self, plugin_sections: list[str] | None = None
    ) -> str:
        """组装决策指令；插件状态区块插在指令之前（None/空白段跳过）。"""
        plugin_block = ""
        cleaned = [
            str(s).strip()
            for s in (plugin_sections or [])
            if s is not None and str(s).strip()
        ]
        if cleaned:
            plugin_block = "【插件状态】\n" + "\n\n".join(cleaned) + "\n\n"
        return f"{plugin_block}{DECISION_DIRECTIVE}"

    # ------------------------------------------------------------------
    # 动态上下文
    # ------------------------------------------------------------------

    def _memory_section(self, reader_name: str) -> str:
        """从 memory profile 读一段文本；异常降级空串。"""
        if self._memory is None:
            return ""
        try:
            reader = getattr(self._memory, reader_name, None)
            if reader is None:
                return ""
            return str(reader() or "").strip()
        except Exception as exc:
            logger.warning("[prompt] memory.%s 读取失败: %s", reader_name, exc)
            return ""

    def read_workspace_context(self) -> str:
        """读取 workspace 主动上下文（截断 + 包装说明）。"""
        if self._workspace_context_fn is None:
            return ""
        try:
            raw = (self._workspace_context_fn() or "").strip()
        except Exception:
            return ""
        if not raw:
            return ""
        return (
            "【工作区主动上下文（主/被动循环共享的规则面板，不是内容源）】\n"
            + raw[:WORKSPACE_CONTEXT_LIMIT]
        )

    def build_context_sections(
        self,
        tick_state: dict[str, Any],
        data: PromptData,
    ) -> list[tuple[str, str]]:
        """组装动态上下文的 (name, content) 段列表（空段跳过）。"""
        sections: list[tuple[str, str]] = []

        fallback_open = bool(tick_state.get("context_as_fallback_open"))
        sections.append((
            "tick_state",
            (
                f"context_fallback={'允许' if fallback_open else '不允许'}\n"
                f"alert_count={len(data.alerts)}\n"
                f"content_count={len(data.content_meta)}\n"
                f"context_count={len(data.context)}"
            ),
        ))

        for name, reader in (
            ("self_model", "read_self"),
            ("long_term_memory", "read_long_term"),
            ("recent_context", "read_recent_context"),
        ):
            content = self._memory_section(reader)
            if content:
                sections.append((name, content))

        alert_block = self.render_alert_block(data.alerts)
        if alert_block:
            sections.append(("proactive_alerts", alert_block))
        content_block = self.render_content_block(
            data.content_meta, data.content_store
        )
        if content_block:
            sections.append(("proactive_content", content_block))
        context_block = self.render_context_block(data.context)
        if context_block:
            sections.append(("proactive_context", context_block))

        workspace = self.read_workspace_context()
        if workspace:
            sections.append(("workspace_proactive_context", workspace))

        return sections

    def build_runtime_context(
        self,
        tick_state: dict[str, Any],
        data: PromptData,
    ) -> str:
        """动态上下文的纯文本形态（调试与测试友好）。"""
        sections = self.build_context_sections(tick_state, data)
        return "\n\n".join(
            f"[{name}]\n{content}" for name, content in sections
        )

    # ------------------------------------------------------------------
    # 三路数据渲染
    # ------------------------------------------------------------------

    @staticmethod
    def render_alert_block(alerts: list[dict[str, Any]]) -> str:
        if not alerts:
            return ""
        lines = []
        for i, item in enumerate(alerts, 1):
            item_id = str(item.get("id") or item.get("item_id") or "")
            title = str(item.get("title") or item.get("content") or "")
            source = str(item.get("source") or "")
            lines.append(f"{i}. [{item_id}] {title}（来源: {source}）")
        return "【Alerts（系统高优通知，优先处理）】\n" + "\n".join(lines)

    @staticmethod
    def render_content_block(
        content_meta: list[dict[str, Any]],
        content_store: dict[str, str],
    ) -> str:
        if not content_meta:
            return ""
        lines = []
        for i, item in enumerate(content_meta, 1):
            item_id = str(item.get("id") or item.get("item_id") or "")
            title = str(item.get("title") or "")
            source = str(item.get("source") or "")
            has_body = "正文已就绪" if content_store.get(item_id) else "正文待获取"
            lines.append(
                f"{i}. [{item_id}] {title}（来源: {source}，{has_body}）"
            )
        return "【Content 候选（正文按需获取）】\n" + "\n".join(lines)

    @staticmethod
    def render_context_block(context: list[dict[str, Any]]) -> str:
        if not context:
            return ""
        payload = json.dumps(context, ensure_ascii=False)[:CONTEXT_JSON_LIMIT]
        return (
            "【背景上下文】\n"
            "注：本段为参考信息，优先级最低；判断时间相关字段时以带 _local "
            "后缀的本地时间为准。\n" + payload
        )


class ProactivePromptModule:
    """prompt 阶段的 phase 模块壳：把组装结果写进 frame.slots。"""

    slot = "prompt"
    phase = "proactive.prompt"

    def __init__(
        self,
        builder: ProactivePromptBuilder,
        *,
        plugin_sections_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self._builder = builder
        self._plugin_sections_fn = plugin_sections_fn

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        plugin_sections = (
            self._plugin_sections_fn() if self._plugin_sections_fn else None
        )
        frame.slots[SLOT_PROMPT_SYSTEM] = self._builder.build_system_prompt(
            plugin_sections
        )

        tick_state = {
            "context_as_fallback_open": frame.slots.get(
                "context_as_fallback_open", False
            ),
        }
        data = PromptData(
            alerts=frame.slots.get("proactive:alerts") or [],
            content_meta=frame.slots.get("proactive:content_meta") or [],
            content_store=frame.slots.get("proactive:content_store") or {},
            context=frame.slots.get("proactive:context") or [],
        )
        frame.slots[SLOT_PROMPT_CONTEXT] = self._builder.build_runtime_context(
            tick_state, data
        )
        return frame
