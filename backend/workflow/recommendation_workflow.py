"""
Recommendation Workflow - LangGraph多智能体工作流
使用LangGraph StateGraph实现多智能体协作流程
"""
from typing import Dict, Any, List, Optional, TypedDict
from loguru import logger

from langgraph.graph import StateGraph, END

from backend.agents.planner_agent import PlannerAgent
from backend.agents.online_search_agent import OnlineSearchAgent
from backend.agents.offline_retrieval_agent import OfflineRetrievalAgent
from backend.agents.recall_agent import RecallAgent
from backend.agents.ranking_agent import RankingAgent
from backend.agents.ui_agent import UIAgent
from backend.services.user_profile import UserProfileService
from backend.services.dedup_filter import DedupFilterService


class RecommendationState(TypedDict):
    """推荐工作流状态"""
    user_id: str
    query: Optional[str]
    user_profile: Dict[str, Any]
    plan: Optional[Dict[str, Any]]
    online_search_results: List[Dict[str, Any]]
    offline_retrieval_results: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    filtered_candidates: List[Dict[str, Any]]
    ranked_papers: List[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    top_n: int
    recall_strategies: Optional[List[str]]
    use_llm_rerank: bool
    error: Optional[str]


class RecommendationWorkflow:
    """Recommendation Workflow using LangGraph StateGraph"""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.online_search = OnlineSearchAgent()
        self.offline_retrieval = OfflineRetrievalAgent()
        self.recall = RecallAgent()
        self.ranking = RankingAgent()
        self.ui = UIAgent()
        self.user_profile_service = UserProfileService()
        self.dedup_filter = DedupFilterService()
        
        # 构建LangGraph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""
        workflow = StateGraph(RecommendationState)
        
        # 添加节点
        workflow.add_node("load_profile", self._load_user_profile)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("online_search", self._online_search_node)
        workflow.add_node("offline_retrieval", self._offline_retrieval_node)
        workflow.add_node("recall", self._recall_node)
        workflow.add_node("dedup_filter", self._dedup_filter_node)
        workflow.add_node("ranking", self._ranking_node)
        workflow.add_node("ui_format", self._ui_format_node)
        workflow.add_node("update_profile", self._update_profile_node)
        
        # 设置入口点
        workflow.set_entry_point("load_profile")
        
        # 定义边（流程）
        workflow.add_edge("load_profile", "planner")
        
        # Planner后的条件路由
        workflow.add_conditional_edges(
            "planner",
            self._should_do_online_search,
            {
                "yes": "online_search",
                "no": "offline_retrieval"
            }
        )
        
        # Online Search后继续到Offline Retrieval
        workflow.add_edge("online_search", "offline_retrieval")
        
        # Offline Retrieval后到Recall
        workflow.add_edge("offline_retrieval", "recall")
        
        # Recall后检查是否有候选
        workflow.add_conditional_edges(
            "recall",
            self._has_candidates,
            {
                "yes": "dedup_filter",
                "no": END
            }
        )
        
        # Dedup Filter后到Ranking
        workflow.add_edge("dedup_filter", "ranking")
        
        # Ranking后到UI格式化
        workflow.add_edge("ranking", "ui_format")
        
        # UI格式化后更新画像
        workflow.add_edge("ui_format", "update_profile")
        
        # 更新画像后结束
        workflow.add_edge("update_profile", END)
        
        return workflow
    
    async def _load_user_profile(self, state: RecommendationState) -> RecommendationState:
        """加载用户画像节点"""
        try:
            user_profile = await self.user_profile_service.get_profile(state["user_id"])
            state["user_profile"] = user_profile
            logger.info(f"Loaded profile for user {state['user_id']}")
        except Exception as e:
            logger.error(f"Failed to load user profile: {e}")
            state["error"] = str(e)
        return state
    
    async def _planner_node(self, state: RecommendationState) -> RecommendationState:
        """Planner规划节点"""
        try:
            plan = await self.planner.plan(
                user_query=state.get("query") or "daily recommendation",
                user_profile=state["user_profile"]
            )
            state["plan"] = plan
            logger.info(f"Planner generated plan: {plan}")
        except Exception as e:
            logger.error(f"Planner error: {e}")
            # 使用默认plan
            state["plan"] = {
                "online_search": True,
                "offline_retrieval": True,
                "recall": True,
                "ranking": True,
                "ui": True
            }
        return state
    
    def _should_do_online_search(self, state: RecommendationState) -> str:
        """判断是否需要在线搜索"""
        plan = state.get("plan", {})
        if plan.get("online_search", False) and state.get("query"):
            return "yes"
        return "no"
    
    async def _online_search_node(self, state: RecommendationState) -> RecommendationState:
        """在线搜索节点"""
        try:
            results = await self.online_search.search(
                query=state.get("query", ""),
                max_results=100
            )
            state["online_search_results"] = results
            logger.info(f"Online search returned {len(results)} papers")
        except Exception as e:
            logger.error(f"Online search error: {e}")
            state["online_search_results"] = []
        return state
    
    async def _offline_retrieval_node(self, state: RecommendationState) -> RecommendationState:
        """离线检索节点"""
        try:
            results = await self.offline_retrieval.retrieve_by_user_profile(
                user_profile=state["user_profile"],
                top_k=200
            )
            state["offline_retrieval_results"] = results
            logger.info(f"Offline retrieval returned {len(results)} papers")
        except Exception as e:
            logger.error(f"Offline retrieval error: {e}")
            state["offline_retrieval_results"] = []
        return state
    
    async def _recall_node(self, state: RecommendationState) -> RecommendationState:
        """召回节点 - 整合多路召回结果"""
        try:
            candidates = await self.recall.recall(
                user_query=state.get("query", ""),
                user_profile=state["user_profile"],
                recall_strategies=state.get("recall_strategies"),
                max_candidates=500
            )
            state["candidates"] = candidates
            logger.info(f"Recall returned {len(candidates)} candidates")
        except Exception as e:
            logger.error(f"Recall error: {e}")
            state["candidates"] = []
        return state
    
    def _has_candidates(self, state: RecommendationState) -> str:
        """判断是否有候选论文"""
        if state.get("candidates") and len(state["candidates"]) > 0:
            return "yes"
        return "no"
    
    async def _dedup_filter_node(self, state: RecommendationState) -> RecommendationState:
        """去重过滤节点"""
        try:
            filtered = await self.dedup_filter.filter(
                papers=state["candidates"],
                user_id=state["user_id"],
                filter_read=True,
                filter_exposed=True
            )
            state["filtered_candidates"] = filtered
            logger.info(f"Filtered to {len(filtered)} candidates")
        except Exception as e:
            logger.error(f"Dedup filter error: {e}")
            state["filtered_candidates"] = state["candidates"]
        return state
    
    async def _ranking_node(self, state: RecommendationState) -> RecommendationState:
        """排序节点"""
        try:
            ranked = await self.ranking.rank(
                candidates=state["filtered_candidates"],
                user_profile=state["user_profile"],
                top_n=state.get("top_n", 20),
                use_llm_rerank=state.get("use_llm_rerank", True)
            )
            state["ranked_papers"] = ranked
            logger.info(f"Ranking returned {len(ranked)} papers")
        except Exception as e:
            logger.error(f"Ranking error: {e}")
            state["ranked_papers"] = state["filtered_candidates"][:state.get("top_n", 20)]
        return state
    
    async def _ui_format_node(self, state: RecommendationState) -> RecommendationState:
        """UI格式化节点"""
        try:
            formatted = await self.ui.format_recommendations(
                papers=state["ranked_papers"],
                user_profile=state["user_profile"],
                format_type="detailed"
            )
            state["result"] = formatted
            logger.info("UI formatting completed")
        except Exception as e:
            logger.error(f"UI format error: {e}")
            state["result"] = {
                "user_id": state["user_id"],
                "total": len(state["ranked_papers"]),
                "papers": state["ranked_papers"],
                "user_profile": state["user_profile"]
            }
        return state
    
    async def _update_profile_node(self, state: RecommendationState) -> RecommendationState:
        """更新用户画像节点"""
        try:
            await self.ui.update_user_profile(
                user_id=state["user_id"],
                recommended_papers=state["ranked_papers"]
            )
            logger.info("User profile updated")
        except Exception as e:
            logger.error(f"Update profile error: {e}")
        return state
    
    async def run(
        self,
        user_id: str,
        query: Optional[str] = None,
        top_n: int = 20,
        recall_strategies: Optional[List[str]] = None,
        use_llm_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        执行推荐工作流（使用LangGraph）
        
        Args:
            user_id: 用户ID
            query: 用户查询（可选）
            top_n: 返回数量
            recall_strategies: 召回策略
            use_llm_rerank: 是否使用LLM重排
            
        Returns:
            推荐结果
        """
        try:
            # 初始化状态
            initial_state: RecommendationState = {
                "user_id": user_id,
                "query": query,
                "user_profile": {},
                "plan": None,
                "online_search_results": [],
                "offline_retrieval_results": [],
                "candidates": [],
                "filtered_candidates": [],
                "ranked_papers": [],
                "result": None,
                "top_n": top_n,
                "recall_strategies": recall_strategies,
                "use_llm_rerank": use_llm_rerank,
                "error": None
            }
            
            logger.info(f"Starting LangGraph workflow for user {user_id}")
            
            # 运行LangGraph
            final_state = await self.app.ainvoke(initial_state)
            
            # 检查错误
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            # 如果没有结果，返回空结果
            if not final_state.get("result"):
                return {
                    "user_id": user_id,
                    "total": 0,
                    "papers": [],
                    "user_profile": final_state.get("user_profile", {})
                }
            
            logger.info(f"LangGraph workflow completed for user {user_id}")
            return final_state["result"]
            
        except Exception as e:
            logger.error(f"LangGraph workflow error: {e}")
            raise
