"""测试生命周期钩子集成"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mind.loop import MindLoop
from mind.extensions.lifecycle import register_hook, unregister_hook


class TestLifecycleHooksIntegration(unittest.IsolatedAsyncioTestCase):
    """测试生命周期钩子在 MindLoop 中的集成"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Path(self.temp_dir)
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_config(self):
        """创建模拟的 Config 对象"""
        mock_config = MagicMock()
        mock_config.workspace = self.workspace
        mock_config.max_history_tokens = 4096
        
        # 设置 runtime mock
        mock_runtime = MagicMock()
        mock_runtime.base_url = "http://localhost:8000"
        mock_runtime.api_key = "test-key"
        mock_runtime.model = "test-model"
        mock_runtime.context_window = 128000
        mock_runtime.max_output_tokens = 4096
        
        mock_config.runtimes = MagicMock()
        mock_config.runtimes.get_main.return_value = mock_runtime
        mock_config.runtimes.get_fast.return_value = None
        
        # 设置 llm mock
        mock_config.llm = mock_runtime
        mock_config.context_compaction = None
        
        return mock_config

    async def test_lifecycle_hooks_called_in_order(self):
        """测试 6 个生命周期钩子按正确顺序调用"""
        call_order = []
        
        def before_turn(ctx):
            call_order.append("before_turn")
            
        def prompt_render(ctx):
            call_order.append("prompt_render")
            
        def before_reasoning(ctx):
            call_order.append("before_reasoning")
            
        def after_reasoning(ctx):
            call_order.append("after_reasoning")
            
        def after_turn(ctx):
            call_order.append("after_turn")
        
        register_hook("before_turn", before_turn)
        register_hook("prompt_render", prompt_render)
        register_hook("before_reasoning", before_reasoning)
        register_hook("after_reasoning", after_reasoning)
        register_hook("after_turn", after_turn)
        
        try:
            mock_config = self._create_mock_config()
            loop = MindLoop(mock_config)
            loop._provider = AsyncMock()
            loop._provider.chat = AsyncMock(return_value=MagicMock(
                content="你好",
                tool_calls=None
            ))
            
            await loop.run("你好")
        
            # 实际调用顺序: before_turn → before_reasoning → prompt_render (在 _build_messages 内) → after_reasoning → after_turn
            expected_order = [
                "before_turn",
                "before_reasoning",
                "prompt_render",
                "after_reasoning",
                "after_turn"
            ]
            self.assertEqual(call_order, expected_order)
            
        finally:
            unregister_hook("before_turn", before_turn)
            unregister_hook("prompt_render", prompt_render)
            unregister_hook("before_reasoning", before_reasoning)
            unregister_hook("after_reasoning", after_reasoning)
            unregister_hook("after_turn", after_turn)

    async def test_before_turn_can_skip_turn(self):
        """测试 before_turn 钩子可以跳过整个 turn"""
        def before_turn(ctx):
            ctx.should_skip = True
            ctx.skip_reason = "被 before_turn 跳过"
        
        register_hook("before_turn", before_turn)
        
        try:
            mock_config = self._create_mock_config()
            loop = MindLoop(mock_config)
            loop._provider = AsyncMock()
            
            response = await loop.run("你好")
            self.assertEqual(response, "被 before_turn 跳过")
            loop._provider.chat.assert_not_called()
            
        finally:
            unregister_hook("before_turn", before_turn)

    async def test_multiple_hooks_same_stage(self):
        """测试同一阶段可以注册多个钩子"""
        call_order = []
        
        def before_turn_1(ctx):
            call_order.append("before_turn_1")
            
        def before_turn_2(ctx):
            call_order.append("before_turn_2")
            
        def before_turn_3(ctx):
            call_order.append("before_turn_3")
        
        register_hook("before_turn", before_turn_1)
        register_hook("before_turn", before_turn_2)
        register_hook("before_turn", before_turn_3)
        
        try:
            mock_config = self._create_mock_config()
            loop = MindLoop(mock_config)
            loop._provider = AsyncMock()
            loop._provider.chat = AsyncMock(return_value=MagicMock(
                content="hi",
                tool_calls=None
            ))
            
            await loop.run("hello")
        
            self.assertEqual(call_order, ["before_turn_1", "before_turn_2", "before_turn_3"])
            
        finally:
            unregister_hook("before_turn", before_turn_1)
            unregister_hook("before_turn", before_turn_2)
            unregister_hook("before_turn", before_turn_3)

    async def test_hooks_in_streaming_mode(self):
        """测试流式输出模式下的生命周期钩子"""
        call_order = []
        
        def before_turn(ctx):
            call_order.append("before_turn")
            
        def after_turn(ctx):
            call_order.append("after_turn")
        
        register_hook("before_turn", before_turn)
        register_hook("after_turn", after_turn)
        
        try:
            mock_config = self._create_mock_config()
            loop = MindLoop(mock_config)
            
            async def mock_stream(*args, **kwargs):
                yield MagicMock(content="你", done=False)
                yield MagicMock(content="好", done=False)
                yield MagicMock(content="", done=True, response=MagicMock(
                    content="你好",
                    tool_calls=None
                ))
            
            loop._provider = AsyncMock()
            loop._provider.chat_stream = mock_stream
            
            chunks = []
            async for chunk in loop.run_stream("hello"):
                chunks.append(chunk)
            
            self.assertIn("before_turn", call_order)
            self.assertIn("after_turn", call_order)
            self.assertEqual("".join(chunks), "你好")
            
        finally:
            unregister_hook("before_turn", before_turn)
            unregister_hook("after_turn", after_turn)


if __name__ == "__main__":
    unittest.main()
