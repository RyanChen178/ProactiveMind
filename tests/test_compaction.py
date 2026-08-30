"""上下文压缩测试。"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from mind.compaction import (
    CompactionCheckpoint,
    ContextCompactor,
    SOFT_LIMIT_RATIO,
)


class CompactionCheckpointTest(unittest.TestCase):
    """测试检查点序列化。"""

    def test_to_dict_and_from_dict(self):
        checkpoint = CompactionCheckpoint(
            generation=1,
            summary="测试摘要",
            context_window=128000,
            soft_limit_tokens=94720,
            keep_recent_tokens=20000,
            estimated_tokens_before=100000,
            estimated_tokens_after=30000,
            compressed_message_count=50,
            retained_message_count=10,
            timestamp="2026-01-01T00:00:00+00:00",
            digest="abc123",
        )
        data = checkpoint.to_dict()
        restored = CompactionCheckpoint.from_dict(data)
        self.assertEqual(restored.generation, 1)
        self.assertEqual(restored.summary, "测试摘要")
        self.assertEqual(restored.compressed_message_count, 50)


class ContextCompactorTest(unittest.TestCase):
    """测试上下文压缩器。"""

    def setUp(self):
        self.mock_provider = MagicMock()
        self.mock_provider.chat = AsyncMock()

    def test_should_compact_below_limit(self):
        """低于软限制时不应压缩。"""
        compactor = ContextCompactor(
            provider=self.mock_provider,
            context_window=128000,
            keep_recent_tokens=20000,
        )
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        self.assertFalse(compactor.should_compact(messages))

    def test_should_compact_above_limit(self):
        """超过软限制时应该压缩。"""
        compactor = ContextCompactor(
            provider=self.mock_provider,
            context_window=1000,
            keep_recent_tokens=200,
        )
        messages = [{"role": "user", "content": "x" * 500} for _ in range(10)]
        self.assertTrue(compactor.should_compact(messages))

    def test_soft_limit_calculation(self):
        """测试软限制计算。"""
        compactor = ContextCompactor(
            provider=self.mock_provider,
            context_window=100000,
            keep_recent_tokens=20000,
        )
        self.assertEqual(compactor.soft_limit, 74000)

    def test_generation_starts_at_zero(self):
        """初始代数为 0。"""
        compactor = ContextCompactor(
            provider=self.mock_provider,
            context_window=128000,
            keep_recent_tokens=20000,
        )
        self.assertEqual(compactor.generation, 0)
        self.assertIsNone(compactor.active_checkpoint)


class ContextCompactorAsyncTest(unittest.IsolatedAsyncioTestCase):
    """测试上下文压缩器的异步方法。"""

    async def test_compact_generates_summary(self):
        """测试压缩生成摘要。"""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "这是对话历史的摘要"
        mock_provider.chat = AsyncMock(return_value=mock_response)

        # 使用很小的 keep_recent_tokens 确保有消息需要压缩
        compactor = ContextCompactor(
            provider=mock_provider,
            context_window=1000,
            keep_recent_tokens=50,
        )

        # 创建大消息确保超过软限制
        messages = [
            {"role": "user", "content": f"这是一条很长的消息 {i} " * 20}
            for i in range(10)
        ]
        compacted, checkpoint = await compactor.compact(messages)

        self.assertTrue(mock_provider.chat.called)
        self.assertEqual(checkpoint.generation, 1)
        self.assertIn("摘要", checkpoint.summary)
        self.assertGreater(len(compacted), 0)

    async def test_compact_preserves_recent_messages(self):
        """测试压缩保留最近的消息。"""
        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "摘要"
        mock_provider.chat = AsyncMock(return_value=mock_response)

        compactor = ContextCompactor(
            provider=mock_provider,
            context_window=1000,
            keep_recent_tokens=50,
        )

        messages = [
            {"role": "user", "content": f"消息内容 {i} " * 10}
            for i in range(10)
        ]
        compacted, checkpoint = await compactor.compact(messages)

        # 应该有摘要消息 + 保留的最近消息
        self.assertGreater(len(compacted), 1)
        total_messages = (
            checkpoint.compressed_message_count + checkpoint.retained_message_count
        )
        self.assertEqual(total_messages, 10)

    async def test_compact_handles_provider_error(self):
        """测试压缩时 provider 出错应降级处理。"""
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(side_effect=Exception("API 错误"))

        compactor = ContextCompactor(
            provider=mock_provider,
            context_window=1000,
            keep_recent_tokens=50,
        )

        messages = [
            {"role": "user", "content": f"消息内容 {i} " * 20}
            for i in range(20)
        ]
        compacted, checkpoint = await compactor.compact(messages)

        self.assertIn("压缩", checkpoint.summary)
        self.assertEqual(checkpoint.generation, 1)


if __name__ == "__main__":
    unittest.main()
