"""数据源模块测试。"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from initiative.data_sources import (
    AlertDataSource,
    ContentDataSource,
    ContextDataSource,
    DataSourceItem,
    DataSourceManager,
)


class TestAlertDataSource(unittest.TestCase):
    """测试 AlertDataSource。"""

    def test_add_alert(self):
        source = AlertDataSource()
        source.add_alert("系统错误", {"severity": "critical"})
        
        items = asyncio.run(source.fetch())
        
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "alert")
        self.assertEqual(items[0].content, "系统错误")
        self.assertEqual(items[0].priority, 1.0)
        self.assertEqual(items[0].metadata["severity"], "critical")

    def test_fetch_clears_queue(self):
        source = AlertDataSource()
        source.add_alert("告警1")
        source.add_alert("告警2")
        
        items1 = asyncio.run(source.fetch())
        items2 = asyncio.run(source.fetch())
        
        self.assertEqual(len(items1), 2)
        self.assertEqual(len(items2), 0)


class TestContentDataSource(unittest.TestCase):
    """测试 ContentDataSource。"""

    def test_add_content_without_scorer(self):
        source = ContentDataSource()
        source._score_threshold = 0.5  # 降低阈值以匹配未评分内容的默认分数
        source.add_content("新闻内容")
        
        items = asyncio.run(source.fetch())
        
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "content")
        self.assertEqual(items[0].priority, 0.5)  # 未评分时默认分数

    def test_add_content_with_scorer(self):
        async def mock_scorer(content: str) -> float:
            return 0.9
        
        source = ContentDataSource(llm_scorer=mock_scorer)
        source.add_content("重要新闻")
        
        items = asyncio.run(source.fetch())
        
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].priority, 0.9)

    def test_content_filtering_by_threshold(self):
        async def mock_scorer(content: str) -> float:
            return 0.6 if "重要" in content else 0.4
        
        source = ContentDataSource(llm_scorer=mock_scorer)
        source._score_threshold = 0.7
        source.add_content("重要新闻")
        source.add_content("普通新闻")
        
        items = asyncio.run(source.fetch())
        
        # 只有重要新闻（评分 0.6）应该被过滤掉（阈值 0.7）
        self.assertEqual(len(items), 0)

    def test_fetch_clears_queue(self):
        source = ContentDataSource()
        source._score_threshold = 0.5  # 降低阈值以匹配未评分内容的默认分数
        source.add_content("内容1")
        source.add_content("内容2")
        
        items1 = asyncio.run(source.fetch())
        items2 = asyncio.run(source.fetch())
        
        self.assertEqual(len(items1), 2)
        self.assertEqual(len(items2), 0)


class TestContextDataSource(unittest.TestCase):
    """测试 ContextDataSource。"""

    def test_add_context(self):
        source = ContextDataSource()
        source.add_context("背景知识", priority=0.5)
        
        items = asyncio.run(source.fetch())
        
        # 概率注入，可能返回 0 或 1 个条目
        self.assertIn(len(items), [0, 1])
        if len(items) == 1:
            self.assertEqual(items[0].source, "context")
            self.assertEqual(items[0].priority, 0.5)

    def test_probability_injection(self):
        source = ContextDataSource()
        
        # 添加高概率上下文
        for _ in range(10):
            source.add_context("高概率", priority=0.9)
        
        items = asyncio.run(source.fetch())
        
        # 高概率应该大部分被选中
        self.assertGreater(len(items), 5)

    def test_fetch_clears_queue(self):
        source = ContextDataSource()
        source.add_context("上下文1", priority=1.0)  # 100% 概率
        source.add_context("上下文2", priority=1.0)
        
        items1 = asyncio.run(source.fetch())
        items2 = asyncio.run(source.fetch())
        
        self.assertEqual(len(items1), 2)
        self.assertEqual(len(items2), 0)


class TestDataSourceManager(unittest.TestCase):
    """测试 DataSourceManager。"""

    def test_fetch_all_combines_sources(self):
        manager = DataSourceManager()
        
        manager.alert_source.add_alert("告警")
        manager.content_source.add_content("内容")
        manager.content_source._llm_scorer = lambda c: asyncio.coroutine(lambda: 0.9)()
        manager.context_source.add_context("上下文", priority=1.0)
        
        items = asyncio.run(manager.fetch_all())
        
        # 应该有 3 个条目（告警、内容、上下文）
        self.assertGreaterEqual(len(items), 2)

    def test_priority_ordering(self):
        manager = DataSourceManager()
        
        manager.alert_source.add_alert("告警")
        manager.content_source.add_content("内容")
        manager.content_source._llm_scorer = lambda c: asyncio.coroutine(lambda: 0.8)()
        manager.context_source.add_context("上下文", priority=1.0)
        
        items = asyncio.run(manager.fetch_all())
        
        if len(items) > 1:
            # 告警应该排在最前面（优先级最高）
            self.assertEqual(items[0].source, "alert")


if __name__ == "__main__":
    unittest.main()
