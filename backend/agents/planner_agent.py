"""
Planner Agent - 决策路由和任务规划
负责根据用户请求和系统状态，决定调用哪些Agent以及执行顺序
"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from backend.config import settings


class PlannerAgent:
    """Planner Agent for routing decisions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
    
    async def plan(
        self,
        user_query: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        规划推荐流程
        
        Args:
            user_query: 用户查询
            user_profile: 用户画像
            context: 上下文信息
            
        Returns:
            规划结果，包含需要调用的agents和执行顺序
        """
        system_prompt = """你是一个智能推荐系统的规划器。根据用户查询和画像，决定需要调用哪些组件：
- online_search: 是否需要在线搜索新论文
- offline_retrieval: 是否需要从向量库检索
- recall: 是否需要召回候选
- ranking: 是否需要排序
- ui: 是否需要生成用户界面

返回JSON格式的规划结果。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户查询: {user_query}\n用户画像: {user_profile}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # TODO: Parse response and return structured plan
            plan = {
                "online_search": True,
                "offline_retrieval": True,
                "recall": True,
                "ranking": True,
                "ui": True,
                "order": ["online_search", "offline_retrieval", "recall", "ranking", "ui"]
            }
            logger.info(f"Planner generated plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Planner error: {e}")
            # Fallback to default plan
            return {
                "online_search": True,
                "offline_retrieval": True,
                "recall": True,
                "ranking": True,
                "ui": True,
                "order": ["offline_retrieval", "recall", "ranking", "ui"]
            }
