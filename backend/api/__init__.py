"""
API Routes
"""
from fastapi import APIRouter
from backend.api import recommendations, user_profile

router = APIRouter()

router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
router.include_router(user_profile.router, prefix="/users", tags=["users"])
