"""自我模型测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from initiative.self_model import SelfModel, SelfModelManager


class SelfModelTest(unittest.TestCase):
    """SelfModel 单元测试。"""
    
    def test_add_preference(self) -> None:
        model = SelfModel()
        model.add_preference("喜欢简洁的代码风格")
        
        self.assertEqual(len(model.preferences), 1)
        self.assertEqual(model.preferences[0], "喜欢简洁的代码风格")
        self.assertIsNotNone(model.updated_at)
    
    def test_add_capability(self) -> None:
        model = SelfModel()
        model.add_capability("擅长 Python 开发")
        
        self.assertEqual(len(model.capabilities), 1)
        self.assertEqual(model.capabilities[0], "擅长 Python 开发")
        self.assertIsNotNone(model.updated_at)
    
    def test_add_goal(self) -> None:
        model = SelfModel()
        model.add_goal("完成项目重构")
        
        self.assertEqual(len(model.goals), 1)
        self.assertEqual(model.goals[0], "完成项目重构")
        self.assertIsNotNone(model.updated_at)
    
    def test_remove_goal(self) -> None:
        model = SelfModel()
        model.add_goal("完成项目重构")
        model.remove_goal("完成项目重构")
        
        self.assertEqual(len(model.goals), 0)
        self.assertIsNotNone(model.updated_at)
    
    def test_to_markdown(self) -> None:
        model = SelfModel()
        model.add_preference("喜欢简洁的代码风格")
        model.add_capability("擅长 Python 开发")
        model.add_goal("完成项目重构")
        
        markdown = model.to_markdown()
        
        self.assertIn("# Self Model", markdown)
        self.assertIn("## Preferences", markdown)
        self.assertIn("- 喜欢简洁的代码风格", markdown)
        self.assertIn("## Capabilities", markdown)
        self.assertIn("- 擅长 Python 开发", markdown)
        self.assertIn("## Goals", markdown)
        self.assertIn("- 完成项目重构", markdown)
    
    def test_from_markdown(self) -> None:
        markdown = """# Self Model

## Preferences

- 喜欢简洁的代码风格
- 偏好函数式编程

## Capabilities

- 擅长 Python 开发

## Goals

- 完成项目重构
"""
        model = SelfModel.from_markdown(markdown)
        
        self.assertEqual(len(model.preferences), 2)
        self.assertIn("喜欢简洁的代码风格", model.preferences)
        self.assertIn("偏好函数式编程", model.preferences)
        self.assertEqual(len(model.capabilities), 1)
        self.assertIn("擅长 Python 开发", model.capabilities)
        self.assertEqual(len(model.goals), 1)
        self.assertIn("完成项目重构", model.goals)
    
    def test_roundtrip(self) -> None:
        """测试 Markdown 序列化/反序列化往返一致性。"""
        original = SelfModel()
        original.add_preference("偏好1")
        original.add_preference("偏好2")
        original.add_capability("能力1")
        original.add_goal("目标1")
        original.add_goal("目标2")
        
        markdown = original.to_markdown()
        restored = SelfModel.from_markdown(markdown)
        
        self.assertEqual(original.preferences, restored.preferences)
        self.assertEqual(original.capabilities, restored.capabilities)
        self.assertEqual(original.goals, restored.goals)


class SelfModelManagerTest(unittest.TestCase):
    """SelfModelManager 单元测试。"""
    
    def test_load_creates_new_model_if_not_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            manager = SelfModelManager(workspace)
            
            model = manager.load()
            
            self.assertIsInstance(model, SelfModel)
            self.assertEqual(len(model.preferences), 0)
            self.assertEqual(len(model.capabilities), 0)
            self.assertEqual(len(model.goals), 0)
    
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            manager = SelfModelManager(workspace)
            
            # 创建并保存模型
            model = SelfModel()
            model.add_preference("测试偏好")
            model.add_capability("测试能力")
            model.add_goal("测试目标")
            manager.save(model)
            
            # 验证文件已创建
            self.assertTrue((workspace / "Self.md").exists())
            
            # 加载并验证
            manager2 = SelfModelManager(workspace)
            loaded = manager2.load()
            
            self.assertEqual(loaded.preferences, ["测试偏好"])
            self.assertEqual(loaded.capabilities, ["测试能力"])
            self.assertEqual(loaded.goals, ["测试目标"])
    
    def test_get_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            manager = SelfModelManager(workspace)
            
            # 第一次调用应该加载或创建
            model1 = manager.get_model()
            self.assertIsInstance(model1, SelfModel)
            
            # 第二次调用应该返回缓存的模型
            model2 = manager.get_model()
            self.assertIs(model1, model2)
    
    def test_update_from_drift_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            manager = SelfModelManager(workspace)
            
            # 初始化模型
            manager.load()
            
            # 模拟 Drift 执行结果
            drift_summary = "学到了一个新的 API 用法"
            manager.update_from_drift(drift_summary)
            
            # 验证能力已添加
            model = manager.get_model()
            self.assertGreater(len(model.capabilities), 0)
            self.assertTrue(any("学到了" in cap for cap in model.capabilities))
    
    def test_update_from_drift_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            manager = SelfModelManager(workspace)
            
            # 初始化模型
            manager.load()
            
            # 模拟 Drift 执行结果
            drift_summary = "需要完成数据库迁移任务"
            manager.update_from_drift(drift_summary)
            
            # 验证目标已添加
            model = manager.get_model()
            self.assertGreater(len(model.goals), 0)
            self.assertTrue(any("需要" in goal for goal in model.goals))


if __name__ == "__main__":
    unittest.main()
