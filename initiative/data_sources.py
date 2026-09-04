"""三路数据源 —— 主动推送的内容供给。

三路数据源设计：
1. Alert   —— 高优先级告警，直接透传（系统错误、安全警告等）
2. Content —— 内容流，LLM 评分分类后决定是否推送
3. Context —— 背景上下文，概率注入（基于相关性评分）
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class DataSourceItem:
    """数据源条目。"""

    source: str  # "alert" | "content" | "context"
    content: str
    priority: float = 0.0  # 0.0-1.0，越高越重要
    metadata: dict[str, Any] | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class DataSource(ABC):
    """数据源基类。"""

    @abstractmethod
    async def fetch(self) -> list[DataSourceItem]:
        """获取数据源中的待推送内容。"""
        ...


class AlertDataSource(DataSource):
    """高优先级告警数据源 —— 直接透传。
    
    用于系统错误、安全警告、紧急通知等必须立即推送的内容。
    所有 alert 类型的条目都会被推送，不进行过滤。
    """

    def __init__(self):
        self._alerts: list[DataSourceItem] = []

    def add_alert(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """添加告警条目。"""
        item = DataSourceItem(
            source="alert",
            content=content,
            priority=1.0,  # 告警始终最高优先级
            metadata=metadata or {"type": "alert"},
        )
        self._alerts.append(item)
        log.info("Added alert: %s", content[:50])

    async def fetch(self) -> list[DataSourceItem]:
        """获取所有告警并清空队列。"""
        alerts = self._alerts.copy()
        self._alerts.clear()
        return alerts


class ContentDataSource(DataSource):
    """内容流数据源 —— LLM 评分分类。
    
    用于文章摘要、新闻、RSS 订阅等内容。
    通过 LLM 对内容进行评分，只有评分高于阈值的内容才会推送。
    """

    def __init__(self, llm_scorer: Callable[[str], Awaitable[float]] | None = None):
        """
        Args:
            llm_scorer: 异步评分函数，输入内容，返回 0.0-1.0 的评分
        """
        self._items: list[DataSourceItem] = []
        self._llm_scorer = llm_scorer
        self._score_threshold = 0.7  # 默认评分阈值

    def add_content(
        self,
        content: str,
        initial_priority: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加内容条目（未经评分）。"""
        item = DataSourceItem(
            source="content",
            content=content,
            priority=initial_priority,
            metadata=metadata or {"type": "content"},
        )
        self._items.append(item)
        log.info("Added content: %s", content[:50])

    async def _score_content(self, content: str) -> float:
        """使用 LLM 对内容进行评分。"""
        if self._llm_scorer is None:
            # 没有评分器时返回默认分数
            return 0.5
        try:
            score = await self._llm_scorer(content)
            return max(0.0, min(1.0, score))  # 限制在 0.0-1.0
        except Exception as exc:
            log.warning("Content scoring failed: %s", exc)
            return 0.5

    async def fetch(self) -> list[DataSourceItem]:
        """获取并评分内容，只返回高于阈值的内容。"""
        scored_items: list[DataSourceItem] = []

        for item in self._items:
            # 对未评分的内容进行评分
            if item.priority == 0.5:  # 初始分数为 0.5 表示未评分
                score = await self._score_content(item.content)
                item.priority = score
                log.info(
                    "Scored content: %.2f - %s",
                    score,
                    item.content[:50],
                )

            # 只返回高于阈值的内容
            if item.priority >= self._score_threshold:
                scored_items.append(item)

        # 清空已处理的内容
        self._items.clear()
        return scored_items


class ContextDataSource(DataSource):
    """背景上下文数据源 —— 概率注入。
    
    用于背景知识、历史上下文等低优先级但有参考价值的信息。
    根据优先级分数进行概率注入：priority=0.8 表示有 80% 概率推送。
    """

    def __init__(self):
        self._items: list[DataSourceItem] = []

    def add_context(
        self,
        content: str,
        priority: float = 0.3,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加上下文条目。
        
        Args:
            content: 上下文内容
            priority: 优先级分数（0.0-1.0），也表示推送概率
            metadata: 元数据
        """
        item = DataSourceItem(
            source="context",
            content=content,
            priority=max(0.0, min(1.0, priority)),
            metadata=metadata or {"type": "context"},
        )
        self._items.append(item)
        log.info("Added context (p=%.2f): %s", priority, content[:50])

    async def fetch(self) -> list[DataSourceItem]:
        """根据优先级概率随机选择推送内容。"""
        selected: list[DataSourceItem] = []

        for item in self._items:
            # 根据优先级进行概率注入
            if random.random() < item.priority:
                selected.append(item)
                log.info(
                    "Selected context (p=%.2f): %s",
                    item.priority,
                    item.content[:50],
                )

        # 清空已处理的上下文
        self._items.clear()
        return selected


class DataSourceManager:
    """数据源管理器 —— 统一管理和调度三路数据源。"""

    def __init__(
        self,
        alert_source: AlertDataSource | None = None,
        content_source: ContentDataSource | None = None,
        context_source: ContextDataSource | None = None,
    ):
        self.alert_source = alert_source or AlertDataSource()
        self.content_source = content_source or ContentDataSource()
        self.context_source = context_source or ContextDataSource()

    async def fetch_all(self) -> list[DataSourceItem]:
        """获取所有数据源的内容并按优先级排序。
        
        返回顺序：alert（最高优先级）> content（中等优先级）> context（最低优先级）
        """
        all_items: list[DataSourceItem] = []

        # 1. Alert: 直接透传
        alerts = await self.alert_source.fetch()
        all_items.extend(alerts)

        # 2. Content: LLM 评分后筛选
        contents = await self.content_source.fetch()
        all_items.extend(contents)

        # 3. Context: 概率注入
        contexts = await self.context_source.fetch()
        all_items.extend(contexts)

        # 按优先级降序排序
        all_items.sort(key=lambda item: item.priority, reverse=True)

        log.info(
            "Fetched %d items: %d alerts, %d contents, %d contexts",
            len(all_items),
            len(alerts),
            len(contents),
            len(contexts),
        )

        return all_items
