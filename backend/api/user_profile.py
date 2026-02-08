"""
User Profile API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.services.user_profile import UserProfileService

router = APIRouter()


class UserFeedbackRequest(BaseModel):
    """用户反馈请求模型"""
    user_id: str
    liked: Optional[List[str]] = []  # Paper IDs
    disliked: Optional[List[str]] = []
    read: Optional[List[str]] = []
    saved: Optional[List[str]] = []


@router.get("/{user_id}/profile")
async def get_user_profile(user_id: str):
    """
    获取用户画像
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户画像
    """
    try:
        service = UserProfileService()
        profile = await service.get_profile(user_id)
        return profile
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/feedback")
async def submit_user_feedback(user_id: str, feedback: UserFeedbackRequest):
    """
    提交用户反馈
    
    Args:
        user_id: 用户ID
        feedback: 用户反馈
        
    Returns:
        更新后的用户画像
    """
    try:
        service = UserProfileService()
        
        # 处理反馈
        if feedback.read:
            await service.add_read_history(user_id, feedback.read)
        
        if feedback.liked:
            await service.update_interests_from_papers(user_id, feedback.liked)
        
        # 更新画像摘要
        updated_profile = await service.generate_profile_summary(user_id)
        
        return {
            "message": "Feedback submitted successfully",
            "profile": updated_profile
        }
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))
