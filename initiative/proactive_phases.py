"""主动链路阶段管线 —— 模块按 phase 注册、依赖排序后链式执行。

设计思想（与 a20b45af proactive_v2/phases.py 及其依赖的
agent/lifecycle/phase.py 中 topo_sort_modules 对齐）：

  - 九个固定阶段（每轮 tick 按此顺序执行）：
      proactive.tick      —— 轮次初始化
      proactive.gate      —— 准入判定
      proactive.source    —— 数据源拉取
      proactive.drift     —— 漂移执行
      proactive.prompt    —— 提示词组装
      proactive.judge     —— 评分决策
      proactive.resolve   —— 冲突消解
      proactive.deliver   —— 投递执行
      proactive.schedule  —— 下轮调度计算
  - 每个模块必须声明 phase（且在合法清单内）与唯一 slot（名字）
  - 同一 phase 内模块可声明 requires（依赖其它 slot）：
      - 依赖缺失的模块被静默禁用（记警告，不报错）
      - 依赖环报 RuntimeError
      - 无依赖约束时按声明顺序稳定执行
  - ProactivePhaseRunner.run(frame)：按阶段顺序把 frame 依次
    传入各模块的 run(frame)，模块返回（可能修改过的）frame
  - inspect()：每个非空 phase 输出执行顺序 + 依赖树文本

ProactiveMind 命名（initiative/proactive_phases.py）独立于原仓库，
排序与树渲染为独立实现。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from initiative.proactive_frame import ProactiveFrame

logger = logging.getLogger(__name__)


PROACTIVE_PHASES: tuple[str, ...] = (
    "proactive.tick",
    "proactive.gate",
    "proactive.source",
    "proactive.drift",
    "proactive.prompt",
    "proactive.judge",
    "proactive.resolve",
    "proactive.deliver",
    "proactive.schedule",
)


def _module_slot(module: object) -> str:
    """读取模块的 slot 声明；缺失或为空抛错。"""
    slot = getattr(module, "slot", None)
    if not isinstance(slot, str) or not slot.strip():
        raise RuntimeError(
            f"模块缺少 slot 声明: {type(module).__name__}"
        )
    return slot.strip()


def _module_requires(module: object, known_slots: set[str]) -> tuple[str, ...]:
    """读取模块的 requires 列表，只保留已知 slot（忽略外部依赖）。"""
    raw = getattr(module, "requires", ()) or ()
    return tuple(
        str(slot) for slot in raw
        if isinstance(slot, str) and slot in known_slots
    )


def _is_builtin(module: object) -> bool:
    """内置模块（声明 builtin=True）不参与缺失依赖禁用。"""
    return bool(getattr(module, "builtin", False))


def _active_slots(modules: Sequence[object]) -> set[str]:
    """迭代剔除依赖缺失的非内置模块（传递性：被禁用后依赖它的也禁用）。

    requires 指向的 slot 不在当前活跃集合（含从未声明的未知 slot）
    即视为缺失——避免依赖被静默丢弃。
    """
    slot_map = {_module_slot(m): m for m in modules}
    active = set(slot_map)
    while True:
        disabled: set[str] = set()
        for slot, module in slot_map.items():
            if slot not in active or _is_builtin(module):
                continue
            declared = {
                str(r) for r in (getattr(module, "requires", ()) or [])
                if isinstance(r, str)
            }
            missing = sorted(declared - active)
            if missing:
                logger.warning(
                    "[phases] 模块依赖不存在，已禁用: module=%s missing=%s",
                    slot,
                    ", ".join(missing),
                )
                disabled.add(slot)
        if not disabled:
            return active
        active -= disabled


def sort_modules(modules: Sequence[object]) -> list[object]:
    """同 phase 内模块拓扑排序（Kahn + 声明顺序稳定 tiebreak）。

    - slot 必须唯一（重复抛错）
    - requires 指向未知 slot 的模块被剔除（由 _active_slots 处理）
    - 依赖环抛 RuntimeError
    """
    slot_map: dict[str, object] = {}
    slot_order: dict[str, int] = {}
    for index, module in enumerate(modules):
        slot = _module_slot(module)
        if slot in slot_map:
            raise RuntimeError(f"模块 slot 重复: {slot}")
        slot_map[slot] = module
        slot_order[slot] = index

    active = _active_slots(modules)
    working = {slot: slot_map[slot] for slot in active if slot in slot_map}

    in_degree = {slot: 0 for slot in working}
    dependents: dict[str, list[str]] = {slot: [] for slot in working}
    for slot, module in working.items():
        for req in _module_requires(module, set(working)):
            in_degree[slot] += 1
            dependents[req].append(slot)

    queue = [slot for slot, degree in in_degree.items() if degree == 0]
    ordered: list[object] = []
    while queue:
        # 内置模块优先，其余按声明顺序（稳定排序）
        queue.sort(key=lambda s: (not _is_builtin(working[s]), slot_order[s]))
        slot = queue.pop(0)
        ordered.append(working[slot])
        for dependent in dependents[slot]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(working):
        unresolved = sorted(s for s, d in in_degree.items() if d > 0)
        raise RuntimeError(f"模块循环依赖: {', '.join(unresolved)}")
    return ordered


def render_dependency_tree(modules: Sequence[object]) -> str:
    """把排序后的模块渲染成 ASCII 依赖树。"""
    ordered = sort_modules(modules)
    slot_map = {_module_slot(m): m for m in ordered}
    children: dict[str, list[str]] = {slot: [] for slot in slot_map}
    in_degree: dict[str, int] = {slot: 0 for slot in slot_map}
    for slot, module in slot_map.items():
        for req in _module_requires(module, set(slot_map)):
            children[req].append(slot)
            in_degree[slot] += 1

    roots = [slot for slot, degree in in_degree.items() if degree == 0]
    roots.sort(key=lambda s: list(slot_map).index(s))
    lines: list[str] = []
    visited: set[str] = set()

    def _render(slot: str, prefix: str, is_last: bool) -> None:
        if slot in visited:
            return
        visited.add(slot)
        connector = "\\-- " if is_last else "|-- "
        lines.append(f"{prefix}{connector}{slot}")
        kids = children[slot]
        child_prefix = prefix + ("    " if is_last else "|   ")
        for i, kid in enumerate(kids):
            _render(kid, child_prefix, i == len(kids) - 1)

    for i, slot in enumerate(roots):
        if i == 0:
            lines.append(slot)
            kids = children[slot]
            for j, kid in enumerate(kids):
                _render(kid, "", j == len(kids) - 1)
        else:
            _render(slot, "", True)
    return "\n".join(lines)


def inspect_modules(modules: Sequence[object]) -> str:
    """输出执行顺序列表 + 依赖树。"""
    ordered = sort_modules(modules)
    chain = "\n".join(
        f"  {i:2d}. {_module_slot(m)}"
        for i, m in enumerate(ordered)
    )
    tree = render_dependency_tree(modules)
    return f"执行顺序:\n{chain}\n\n依赖树:\n{tree}"


class ProactivePhaseRunner:
    """按阶段分组并链式执行主动链路模块。"""

    def __init__(self, modules: Iterable[object]) -> None:
        grouped: dict[str, list[object]] = {phase: [] for phase in PROACTIVE_PHASES}
        for module in modules:
            phase = getattr(module, "phase", None)
            if not isinstance(phase, str) or not phase:
                raise RuntimeError(
                    f"主动链路模块缺少 phase 声明: {type(module).__name__}"
                )
            if phase not in PROACTIVE_PHASES:
                raise RuntimeError(f"未知主动链路 phase: {phase}")
            grouped[phase].append(module)
        self._modules_by_phase: dict[str, list[object]] = {
            phase: sort_modules(grouped[phase])
            for phase in PROACTIVE_PHASES
        }

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        """按阶段顺序链式执行所有模块；模块返回的 frame 继续传递。"""
        for phase in PROACTIVE_PHASES:
            for module in self._modules_by_phase[phase]:
                frame = await module.run(frame)  # type: ignore[attr-defined]
        return frame

    def inspect(self) -> str:
        sections: list[str] = []
        for phase in PROACTIVE_PHASES:
            modules = self._modules_by_phase[phase]
            if not modules:
                continue
            sections.append(f"[{phase}]\n{inspect_modules(modules)}")
        return "\n\n".join(sections)

    @property
    def modules_by_phase(self) -> dict[str, list[object]]:
        """每个阶段的模块列表（拷贝）。"""
        return {
            phase: list(modules)
            for phase, modules in self._modules_by_phase.items()
        }

    def all_slots(self) -> dict[str, list[str]]:
        """每个阶段的 slot 名列表（调试用）。"""
        return {
            phase: [_module_slot(m) for m in modules]
            for phase, modules in self._modules_by_phase.items()
        }
