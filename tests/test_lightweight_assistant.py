"""轻量模型辅助模块的单元测试"""

import pytest
from unittest.mock import AsyncMock, Mock
from mind.lightweight_assistant import LightweightModelAssistant


class TestLightweightModelAssistant:
    """测试轻量模型辅助器"""
    
    def test_init_with_provider(self):
        """测试初始化时传入 provider"""
        mock_provider = Mock()
        assistant = LightweightModelAssistant(mock_provider)
        assert assistant.available is True
    
    def test_init_without_provider(self):
        """测试初始化时不传入 provider"""
        assistant = LightweightModelAssistant(None)
        assert assistant.available is False
    
    @pytest.mark.asyncio
    async def test_memory_gate_filter_facts(self):
        """测试记忆门控过滤事实"""
        # 准备 mock provider
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = """- 用户喜欢编程
- 用户住在北京"""
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        assistant = LightweightModelAssistant(mock_provider)
        
        # 测试过滤
        user_input = "今天天气不错"
        assistant_reply = "是的，天气很好"
        existing_facts = [
            "用户喜欢编程",
            "今天天气不错",
            "用户住在北京"
        ]
        
        result = await assistant.memory_gate(user_input, assistant_reply, existing_facts)
        
        # 验证结果
        assert len(result) == 2
        assert "用户喜欢编程" in result
        assert "用户住在北京" in result
        assert "今天天气不错" not in result
        
        # 验证调用了 provider
        mock_provider.chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_gate_no_provider(self):
        """测试记忆门控在没有 provider 时的降级行为"""
        assistant = LightweightModelAssistant(None)
        
        user_input = "用户输入"
        assistant_reply = "助手回复"
        existing_facts = ["事实1", "事实2"]
        
        result = await assistant.memory_gate(user_input, assistant_reply, existing_facts)
        
        # 降级时返回所有事实
        assert result == existing_facts
    
    @pytest.mark.asyncio
    async def test_memory_gate_empty_facts(self):
        """测试记忆门控处理空事实列表"""
        mock_provider = Mock()
        assistant = LightweightModelAssistant(mock_provider)
        
        result = await assistant.memory_gate("输入", "回复", [])
        assert result == []
        
        result = await assistant.memory_gate("输入", "回复", None)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_query_rewrite(self):
        """测试查询改写"""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "请帮我写一个 Python 的快速排序算法实现"
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        assistant = LightweightModelAssistant(mock_provider)
        
        user_query = "快速排序"
        result = await assistant.query_rewrite(user_query)
        
        # 验证改写结果
        assert result == "请帮我写一个 Python 的快速排序算法实现"
        mock_provider.chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_rewrite_no_provider(self):
        """测试查询改写在没有 provider 时的降级行为"""
        assistant = LightweightModelAssistant(None)
        
        user_query = "原始查询"
        result = await assistant.query_rewrite(user_query)
        
        # 降级时返回原始查询
        assert result == user_query
    
    @pytest.mark.asyncio
    async def test_query_rewrite_short_query(self):
        """测试查询改写处理过短的查询"""
        mock_provider = Mock()
        assistant = LightweightModelAssistant(mock_provider)
        
        # 测试空字符串
        result = await assistant.query_rewrite("")
        assert result == ""
        
        # 测试太短的字符串
        result = await assistant.query_rewrite("ab")
        assert result == "ab"
        
        # 验证没有调用 provider
        mock_provider.chat.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_query_rewrite_empty_response(self):
        """测试查询改写处理空响应"""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = ""
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        assistant = LightweightModelAssistant(mock_provider)
        
        user_query = "原始查询"
        result = await assistant.query_rewrite(user_query)
        
        # 空响应时返回原始查询
        assert result == user_query
    
    @pytest.mark.asyncio
    async def test_deduplicate_facts(self):
        """测试事实去重"""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = """1
3"""
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        assistant = LightweightModelAssistant(mock_provider)
        
        new_facts = [
            "用户喜欢 Python",
            "用户喜欢编程",  # 重复
            "用户住在北京"
        ]
        existing_facts = [
            "用户喜欢编程",
            "用户是一名工程师"
        ]
        
        result = await assistant.deduplicate_facts(new_facts, existing_facts)
        
        # 验证去重结果
        assert len(result) == 2
        assert "用户喜欢 Python" in result
        assert "用户住在北京" in result
        assert "用户喜欢编程" not in result
        
        mock_provider.chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deduplicate_facts_no_provider(self):
        """测试事实去重在没有 provider 时的降级行为"""
        assistant = LightweightModelAssistant(None)
        
        new_facts = ["事实1", "事实2"]
        existing_facts = ["已有事实"]
        
        result = await assistant.deduplicate_facts(new_facts, existing_facts)
        
        # 降级时返回所有新事实
        assert result == new_facts
    
    @pytest.mark.asyncio
    async def test_deduplicate_facts_empty_new(self):
        """测试事实去重处理空的新事实列表"""
        mock_provider = Mock()
        assistant = LightweightModelAssistant(mock_provider)
        
        result = await assistant.deduplicate_facts([], ["已有事实"])
        assert result == []
        
        mock_provider.chat.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_deduplicate_facts_empty_existing(self):
        """测试事实去重处理空的已有事实列表"""
        mock_provider = Mock()
        assistant = LightweightModelAssistant(mock_provider)
        
        new_facts = ["事实1", "事实2"]
        result = await assistant.deduplicate_facts(new_facts, [])
        
        # 没有已有事实时返回所有新事实
        assert result == new_facts
        
        mock_provider.chat.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_deduplicate_facts_all_duplicates(self):
        """测试事实去重处理所有事实都重复的情况"""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = ""
        mock_provider.chat = AsyncMock(return_value=mock_response)
        
        assistant = LightweightModelAssistant(mock_provider)
        
        new_facts = ["事实1", "事实2"]
        existing_facts = ["事实1", "事实2"]
        
        result = await assistant.deduplicate_facts(new_facts, existing_facts)
        
        # 所有事实都重复时返回空列表
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
