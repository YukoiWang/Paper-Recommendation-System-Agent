"""
Recommendations API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.workflow.recommendation_workflow import RecommendationWorkflow

router = APIRouter()


class RecommendationRequest(BaseModel):
    """推荐请求模型"""
    user_id: str
    query: Optional[str] = None
    top_n: int = 20
    recall_strategies: Optional[List[str]] = None
    use_llm_rerank: bool = True


class RecommendationResponse(BaseModel):
    """推荐响应模型"""
    user_id: str
    total: int
    papers: List[Dict[str, Any]]
    user_profile: Dict[str, Any]


@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    获取论文推荐
    
    Args:
        request: 推荐请求
        
    Returns:
        推荐结果
    """
    try:
        workflow = RecommendationWorkflow()
        result = await workflow.run(
            user_id=request.user_id,
            query=request.query,
            top_n=request.top_n,
            recall_strategies=request.recall_strategies,
            use_llm_rerank=request.use_llm_rerank
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            total=result["total"],
            papers=result["papers"],
            user_profile=result["user_profile"]
        )
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily/{user_id}")
async def get_daily_recommendations(
    user_id: str,
    top_n: int = Query(20, ge=1, le=100)
):
    """
    获取每日推荐
    
    Args:
        user_id: 用户ID
        top_n: 返回数量
        
    Returns:
        每日推荐结果
    """
    try:
        workflow = RecommendationWorkflow()
        result = await workflow.run(
            user_id=user_id,
            query=None,  # 每日推荐不需要查询
            top_n=top_n
        )
        
        return result
    except Exception as e:
        logger.error(f"Daily recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
