"""主动链路内核 —— 全部积木的收口：生命周期 + 单轮 tick。

设计思想（与 a20b45af agent/core/proactive_kernel.py 对齐）：

  - ProactiveKernel 把九阶段管线包装成一个可独立启动/停止/
    单轮执行的完整对象：
      start()     —— 顺序调用各模块的可选 start()（如后台轮询源）
      stop()      —— 逆序调用各模块的可选 stop()（与启动相反的
                     顺序释放，依赖关系正确回收）
      run_tick(session_key) —— 一轮：
          1. gateway 预取（可选注入，DataGateway.run() 的三路
             快照并入 initial slots，source 模块可直接消费）
          2. initial_slots_fn(session_key) 注入额外初始槽位
          3. new_frame → ProactivePhaseRunner.run 串通九阶段
          4. 返回 frame.output.base_score（无输出返回 None）
      inspect()   —— 管线的结构描述（调试用）
  - 模块不强制实现 start/stop（getattr 探测，缺省跳过）
  - 与 ProactiveLoop 的分工：kernel 负责"一轮做什么"，
    loop 负责"什么时候做一轮"

复用 initiative.proactive_phases.ProactivePhaseRunner 与
initiative.proactive_gateway.DataGateway。

ProactiveMind 命名（initiative/proactive_kernel.py）独立于原仓库。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable

from initiative.proactive_frame import ProactiveTickResult, new_frame
from initiative.proactive_gateway import DataGateway
from initiative.proactive_phases import ProactivePhaseRunner

logger = logging.getLogger(__name__)

# gateway 预取结果并入 initial slots 的键名
GATEWAY_SLOTS_PREFIX = "proactive:"


class ProactiveKernel:
    """主动链路内核：模块生命周期 + 单轮 tick 收口。"""

    def __init__(
        self,
        modules: Iterable[object],
        *,
        initial_slots_fn: Callable[[str], dict[str, Any]] | None = None,
        gateway: DataGateway | None = None,
    ) -> None:
        self._runner = ProactivePhaseRunner(modules)
        self._initial_slots_fn = initial_slots_fn
        self._gateway = gateway
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """顺序启动所有声明了 start() 的模块。"""
        for module in self._all_modules():
            starter = getattr(module, "start", None)
            if starter is not None:
                await starter()
        self._started = True
        logger.info("[kernel] 已启动 modules=%d", len(self._all_modules()))

    async def stop(self) -> None:
        """逆序停止所有声明了 stop() 的模块（与启动相反的顺序）。"""
        for module in reversed(self._all_modules()):
            stopper = getattr(module, "stop", None)
            if stopper is not None:
                await stopper()
        self._started = False
        logger.info("[kernel] 已停止")

    async def run_tick(self, session_key: str) -> float | None:
        """执行一轮主动链路，返回 base_score（无输出为 None）。"""
        slots: dict[str, Any] = {}

        # 1. gateway 预取（可选）——三路快照并入 initial slots
        if self._gateway is not None:
            snapshot = await self._gateway.run()
            slots[f"{GATEWAY_SLOTS_PREFIX}alerts"] = snapshot.alerts
            slots[f"{GATEWAY_SLOTS_PREFIX}context"] = snapshot.context
            slots[f"{GATEWAY_SLOTS_PREFIX}content_meta"] = snapshot.content_meta
            slots[f"{GATEWAY_SLOTS_PREFIX}content_store"] = dict(
                snapshot.content_store
            )

        # 2. 额外初始槽位（与 gateway 键冲突时以后者覆盖为准的
        #    次序：先注入自定义，再由 gateway 快照覆盖同名键）
        if self._initial_slots_fn is not None:
            try:
                custom = self._initial_slots_fn(session_key) or {}
                merged = dict(custom)
                merged.update(slots)
                slots = merged
            except Exception as exc:
                logger.warning("[kernel] initial_slots 注入失败: %s", exc)

        # 3. 串通九阶段
        frame = await self._runner.run(new_frame(session_key, slots))
        if frame.output is None:
            return None
        return frame.output.base_score

    def inspect(self) -> str:
        """管线的结构描述（阶段 + 顺序 + 依赖树）。"""
        return self._runner.inspect()

    def _all_modules(self) -> list[Any]:
        modules: list[Any] = []
        for phase_modules in self._runner.modules_by_phase.values():
            modules.extend(phase_modules)
        return modules
