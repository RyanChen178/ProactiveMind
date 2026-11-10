"""主动链路总装配 —— 把全部积木组装成可运行的 kernel + loop。

设计思想（与 a20b45af 的 agent_tick_factory / bootstrap/proactive.py
总装配语义对齐，但自包含、依赖注入）：

  - ProactiveEngineDeps：收集全部外部依赖（可选居多，
    缺省时对应能力自动缺席，不阻断装配）
  - build_proactive_engine(cfg, deps)：
      1. 组装模块清单（按需添加，各有缺席条件）：
         DayNightGateModule（gate 阶段）
         EmotionPromptModule（prompt 阶段，情绪库路径给出才加）
         FeedbackStatsModule（prompt 阶段，反馈库路径给出才加）
         ProactiveSourceModule（source 阶段，data_sources 给出才加）
         ProactiveScheduleModule（schedule 阶段，恒装配）
         外部 extra_modules 追加（judge/resolve/deliver 等由
         调用方按 LLM 可用性注入）
      2. gateway（给出三路取数函数才建，否则 None）
      3. kernel = ProactiveKernel(modules, gateway=gateway)
      4. scheduler = ProactiveScheduler(cfg, presence)
      5. loop = ProactiveLoop(runner, scheduler, session_key)
  - 返回 (kernel, loop)，调用方负责 kernel.start() → loop.run()
    → loop.stop() → kernel.stop() 的生命周期

复用全部已有积木；ProactiveMind 命名
（initiative/proactive_builder.py）独立于原仓库。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from initiative.proactive_daynight_module import DayNightConfig, DayNightGateModule
from initiative.proactive_emotion_module import EmotionPromptModule
from initiative.proactive_feedback_module import FeedbackStatsModule
from initiative.proactive_gateway import DataGateway
from initiative.proactive_kernel import ProactiveKernel
from initiative.proactive_loop import ProactiveLoop
from initiative.proactive_phases import ProactivePhaseRunner
from initiative.proactive_schedule import ProactiveScheduler, ProactiveScheduleModule
from initiative.proactive_source import ProactiveSource, ProactiveSourceModule

logger = logging.getLogger(__name__)


@dataclass
class ProactiveEngineDeps:
    """主动链路装配所需的外部依赖（全部可选，缺省即缺席）。"""

    presence: Any = None
    data_sources: Any = None  # DataSourceManager 式三路容器
    emotion_db_path: Path | None = None
    feedback_db_path: Path | None = None
    daynight_config: DayNightConfig | None = None
    gateway: DataGateway | None = None
    extra_modules: list[Any] = field(default_factory=list)


def build_proactive_engine(
    cfg: Any,
    deps: ProactiveEngineDeps | None = None,
) -> tuple[ProactiveKernel, ProactiveLoop]:
    """组装 (kernel, loop)。

    cfg 提供 channel / chat_id 与调度参数（duck typing，
    ProactiveConfig 或等价对象均可）。
    """
    deps = deps or ProactiveEngineDeps()
    modules: list[Any] = []

    # gate 阶段：昼夜静默（配置给出或默认启用）
    modules.append(
        DayNightGateModule(deps.daynight_config or DayNightConfig())
    )

    # source 阶段：三路取数（容器给出才装配）
    if deps.data_sources is not None:
        content_limit = int(
            getattr(cfg, "agent_tick_content_limit", 5) or 5
        )
        modules.append(ProactiveSourceModule(
            ProactiveSource(deps.data_sources, content_limit=content_limit)
        ))

    # prompt 阶段：情绪语气 + 反馈统计（各自库路径给出才装配）
    if deps.emotion_db_path is not None:
        modules.append(EmotionPromptModule(deps.emotion_db_path))
    if deps.feedback_db_path is not None:
        modules.append(FeedbackStatsModule(deps.feedback_db_path))

    # 调用方注入的额外模块（judge / resolve / deliver 等）
    modules.extend(deps.extra_modules)

    # schedule 阶段：恒装配
    scheduler = ProactiveScheduler(cfg=cfg, presence=deps.presence)
    modules.append(ProactiveScheduleModule(scheduler))

    kernel = ProactiveKernel(modules, gateway=deps.gateway)
    runner = ProactivePhaseRunner(modules)

    session_key = _session_key(cfg)
    loop = ProactiveLoop(
        runner=runner,
        scheduler=scheduler,
        session_key=session_key,
    )
    logger.info(
        "[builder] 装配完成 modules=%d session=%s gateway=%s",
        len(modules), session_key or "-", deps.gateway is not None,
    )
    return kernel, loop


def _session_key(cfg: Any) -> str:
    """cfg 的 channel + chat_id 组合会话键。"""
    channel = str(getattr(cfg, "default_channel", "") or "").strip()
    chat_id = str(getattr(cfg, "default_chat_id", "") or "").strip()
    if not channel or not chat_id:
        return ""
    return f"{channel}:{chat_id}"
