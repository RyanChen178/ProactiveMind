"""健康检查 —— 运行时状态检查。

HealthChecker 收集各组件状态，返回整体健康报告。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any


@dataclass
class ComponentHealth:
    """单个组件的健康状态。"""

    name: str
    healthy: bool
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthReport:
    """整体健康报告。"""

    status: str  # "healthy" | "degraded" | "unhealthy"
    timestamp: str
    components: list[ComponentHealth] = field(default_factory=list)

    @property
    def all_healthy(self) -> bool:
        return all(c.healthy for c in self.components)


class HealthChecker:
    """健康检查器。"""

    def __init__(self) -> None:
        self._checks: dict[str, callable] = {}

    def register(self, name: str, check_fn: callable) -> None:
        """注册一个组件检查函数。

        check_fn 返回 (healthy: bool, detail: str)。
        """
        self._checks[name] = check_fn

    def check(self) -> HealthReport:
        """执行所有检查，返回健康报告。"""
        components: list[ComponentHealth] = []

        for name, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                healthy, detail = check_fn()
                if not isinstance(healthy, bool):
                    healthy = False
                    detail = f"检查函数返回了非布尔值: {type(healthy)}"
            except Exception as exc:
                healthy = False
                detail = str(exc)
            latency_ms = (time.monotonic() - start) * 1000
            components.append(
                ComponentHealth(
                    name=name,
                    healthy=healthy,
                    detail=detail,
                    latency_ms=round(latency_ms, 2),
                )
            )

        if not components:
            status = "healthy"
        elif all(c.healthy for c in components):
            status = "healthy"
        elif any(c.healthy for c in components):
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthReport(
            status=status,
            timestamp=datetime.now(UTC).isoformat(),
            components=components,
        )

    def to_dict(self, report: HealthReport) -> dict[str, Any]:
        """将报告转为 dict（用于 JSON 响应）。"""
        return {
            "status": report.status,
            "timestamp": report.timestamp,
            "components": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "detail": c.detail,
                    "latency_ms": c.latency_ms,
                }
                for c in report.components
            ],
        }


def create_health_checker(
    memory_store=None,
    session_store=None,
    presence_store=None,
) -> HealthChecker:
    """创建默认健康检查器，检查核心组件状态。"""
    checker = HealthChecker()

    if memory_store is not None:
        def check_memory():
            try:
                memory_store.read_all()
                return True, "ok"
            except Exception as exc:
                return False, str(exc)
        checker.register("memory", check_memory)

    if session_store is not None:
        def check_sessions():
            try:
                session_store.get_or_create_active_session()
                return True, "ok"
            except Exception as exc:
                return False, str(exc)
        checker.register("sessions", check_sessions)

    if presence_store is not None:
        def check_presence():
            try:
                presence_store.get_last_user_at()
                return True, "ok"
            except Exception as exc:
                return False, str(exc)
        checker.register("presence", check_presence)

    return checker
