"""轻量模型辅助模块 - 使用快速模型进行记忆门控、查询改写和事实去重"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mind.provider import LLMProvider

logger = logging.getLogger(__name__)


class LightweightModelAssistant:
    """使用轻量级快速模型执行辅助任务"""
    
    def __init__(self, fast_provider: LLMProvider | None):
        """初始化轻量模型辅助器
        
        Args:
            fast_provider: 快速模型提供者，如果为 None 则所有功能降级
        """
        self._fast_provider = fast_provider
        self._available = fast_provider is not None
    
    @property
    def available(self) -> bool:
        """检查轻量模型是否可用"""
        return self._available
    
    async def memory_gate(
        self,
        user_input: str,
        assistant_reply: str,
        existing_facts: list[str] | None = None,
    ) -> list[str]:
        """记忆门控：判断哪些提取的事实值得保存"""
        if not self._available:
            logger.debug("轻量模型不可用，记忆门控降级")
            return existing_facts or []
        
        if not existing_facts:
            return []
        
        # 构建评估提示
        facts_text = "\n".join(f"- {fact}" for fact in existing_facts)
        prompt = f"""评估以下从对话中提取的事实，判断哪些值得长期保存。

对话上下文：
用户：{user_input}
助手：{assistant_reply}

待评估的事实：
{facts_text}

请分析每条事实，输出值得保留的事实（保持原格式），并删除以下类型的内容：
1. 临时性信息（如"今天天气不错"）
2. 已经过时的信息
3. 过于琐碎的细节
4. 与对话主题无关的随机信息

只输出值得保留的事实，每行一条，以"- "开头。如果所有事实都不值得保留，输出空。"""

        try:
            response = await self._fast_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            
            # 解析返回的事实
            result_facts = []
            for line in response.content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    result_facts.append(line[2:].strip())
            
            filtered_count = len(existing_facts) - len(result_facts)
            if filtered_count > 0:
                logger.info(f"记忆门控过滤了 {filtered_count} 条事实")
            
            return result_facts
            
        except Exception as e:
            logger.warning(f"记忆门控调用失败: {e}，降级返回所有事实")
            return existing_facts
    
    async def query_rewrite(self, user_query: str) -> str:
        """查询改写：优化用户查询以提高理解质量"""
        if not self._available:
            return user_query
        
        if not user_query or len(user_query.strip()) < 3:
            return user_query
        
        prompt = f"""请优化以下用户查询，使其更清晰、更具体，便于助手理解：

原始查询：{user_query}

优化原则：
1. 如果查询已经很清晰，保持不变
2. 如果有歧义，添加必要的上下文
3. 如果过于简短，适当扩展（但不要添加不存在的信息）
4. 保持用户的原始意图

只输出优化后的查询，不要解释。"""

        try:
            response = await self._fast_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            
            rewritten = response.content.strip()
            
            if not rewritten or len(rewritten) < 3:
                return user_query
            
            if rewritten != user_query:
                logger.debug(f"查询改写: \x27{user_query}\x27 -> \x27{rewritten}\x27")
            
            return rewritten
            
        except Exception as e:
            logger.warning(f"查询改写失败: {e}")
            return user_query
    
    async def deduplicate_facts(
        self,
        new_facts: list[str],
        existing_facts: list[str],
    ) -> list[str]:
        """事实去重：过滤掉与已有事实重复或语义相似的新事实"""
        if not self._available:
            return new_facts
        
        if not new_facts:
            return []
        
        if not existing_facts:
            return new_facts
        
        # 构建去重提示
        new_text = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(new_facts))
        existing_text = "\n".join(f"- {fact}" for fact in existing_facts[:20])
        
        prompt = f"""判断以下新事实中，哪些与已有事实重复或语义相似。

已有事实：
{existing_text}

新事实：
{new_text}

请输出不重复的新事实编号（如：1, 3, 5），每行一个数字。
如果所有新事实都重复，输出空。"""

        try:
            response = await self._fast_provider.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            
            # 解析返回的编号
            unique_indices = set()
            for line in response.content.split("\n"):
                line = line.strip()
                if line.isdigit():
                    idx = int(line) - 1
                    if 0 <= idx < len(new_facts):
                        unique_indices.add(idx)
            
            result = [new_facts[i] for i in sorted(unique_indices)]
            filtered_count = len(new_facts) - len(result)
            
            if filtered_count > 0:
                logger.info(f"事实去重过滤了 {filtered_count} 条重复事实")
            
            return result
            
        except Exception as e:
            logger.warning(f"事实去重失败: {e}，返回所有新事实")
            return new_facts

