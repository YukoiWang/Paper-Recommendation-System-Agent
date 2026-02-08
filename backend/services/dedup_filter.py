"""
Deduplication Filter Service
去重过滤服务（已读/已曝光过滤）
"""
from typing import List, Dict, Any
from loguru import logger

from backend.services.user_profile import UserProfileService


class DedupFilterService:
    """Deduplication and Filter Service"""
    
    def __init__(self):
        self.user_profile_service = UserProfileService()
    
    async def filter(
        self,
        papers: List[Dict[str, Any]],
        user_id: str,
        filter_read: bool = True,
        filter_exposed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        过滤已读和已曝光的论文
        
        Args:
            papers: 论文列表
            user_id: 用户ID
            filter_read: 是否过滤已读论文
            filter_exposed: 是否过滤已曝光论文
            
        Returns:
            过滤后的论文列表
        """
        if not papers:
            return []
        
        # 获取用户历史
        profile = await self.user_profile_service.get_profile(user_id)
        read_history = set(profile.get("read_history", []))
        exposure_history = set(profile.get("exposure_history", []))
        
        filtered_papers = []
        filtered_count = 0
        
        for paper in papers:
            paper_id = paper.get("paper_id") or paper.get("id")
            
            # 过滤已读
            if filter_read and paper_id in read_history:
                filtered_count += 1
                continue
            
            # 过滤已曝光
            if filter_exposed and paper_id in exposure_history:
                filtered_count += 1
                continue
            
            filtered_papers.append(paper)
        
        logger.info(
            f"Filtered {filtered_count} papers (read: {filter_read}, exposed: {filter_exposed}), "
            f"remaining: {len(filtered_papers)}"
        )
        
        return filtered_papers
