"""模型能力目录测试。"""

import unittest

from mind.catalog import (
    ModelCapability,
    lookup_model_capability,
    get_context_window,
    get_max_output_tokens,
    supports_vision,
)


class TestModelCapability(unittest.TestCase):
    """测试 ModelCapability 数据类。"""
    
    def test_to_dict(self):
        """测试 to_dict 方法。"""
        capability = ModelCapability(
            context_window=128000,
            supports_vision=True,
            max_output_tokens=4096,
        )
        result = capability.to_dict()
        self.assertEqual(result["context_window"], 128000)
        self.assertEqual(result["supports_vision"], True)
        self.assertEqual(result["max_output_tokens"], 4096)


class TestLookupModelCapability(unittest.TestCase):
    """测试模型能力查找。"""
    
    def test_lookup_known_model(self):
        """测试查找已知模型。"""
        capability = lookup_model_capability("deepseek", "deepseek-chat")
        self.assertIsNotNone(capability)
        self.assertEqual(capability.context_window, 64000)
        self.assertFalse(capability.supports_vision)
    
    def test_lookup_vision_model(self):
        """测试查找支持视觉的模型。"""
        capability = lookup_model_capability("openai", "gpt-4o")
        self.assertIsNotNone(capability)
        self.assertTrue(capability.supports_vision)
    
    def test_lookup_unknown_model(self):
        """测试查找未知模型返回 None。"""
        capability = lookup_model_capability("unknown", "unknown-model")
        self.assertIsNone(capability)
    
    def test_lookup_by_model_name_only(self):
        """测试只通过模型名查找。"""
        capability = lookup_model_capability("any-provider", "gpt-4")
        self.assertIsNotNone(capability)
        self.assertEqual(capability.context_window, 8192)


class TestGetContextWindow(unittest.TestCase):
    """测试获取上下文窗口。"""
    
    def test_known_model(self):
        """测试获取已知模型的上下文窗口。"""
        window = get_context_window("deepseek", "deepseek-v3")
        self.assertEqual(window, 128000)
    
    def test_unknown_model_with_default(self):
        """测试获取未知模型的上下文窗口（使用默认值）。"""
        window = get_context_window("unknown", "unknown-model", default=32000)
        self.assertEqual(window, 32000)
    
    def test_unknown_model_with_default_128k(self):
        """测试获取未知模型的上下文窗口（默认 128k）。"""
        window = get_context_window("unknown", "unknown-model")
        self.assertEqual(window, 128000)


class TestGetMaxOutputTokens(unittest.TestCase):
    """测试获取最大输出 token 数。"""
    
    def test_model_with_max_output(self):
        """测试获取有 max_output_tokens 的模型。"""
        # 大多数模型在目录中没有设置 max_output_tokens
        max_tokens = get_max_output_tokens("deepseek", "deepseek-chat")
        self.assertEqual(max_tokens, 0)  # 默认为 0（不限制）
    
    def test_unknown_model_with_default(self):
        """测试获取未知模型的最大输出 token 数（使用默认值）。"""
        max_tokens = get_max_output_tokens("unknown", "unknown-model", default=4096)
        self.assertEqual(max_tokens, 4096)


class TestSupportsVision(unittest.TestCase):
    """测试视觉支持检查。"""
    
    def test_vision_model(self):
        """测试支持视觉的模型。"""
        self.assertTrue(supports_vision("openai", "gpt-4o"))
        self.assertTrue(supports_vision("anthropic", "claude-3-opus"))
    
    def test_non_vision_model(self):
        """测试不支持视觉的模型。"""
        self.assertFalse(supports_vision("deepseek", "deepseek-chat"))
        self.assertFalse(supports_vision("openai", "gpt-3.5-turbo"))
    
    def test_unknown_model(self):
        """测试未知模型默认不支持视觉。"""
        self.assertFalse(supports_vision("unknown", "unknown-model"))


if __name__ == "__main__":
    unittest.main()
