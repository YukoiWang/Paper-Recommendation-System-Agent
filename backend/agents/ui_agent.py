"""
User Interface Agent - 用户界面生成
生成推荐结果的展示格式，更新用户画像
"""
from typing import List, Dict, Any
from loguru import logger

from backend.services.user_profile import UserProfileService
from backend.services.llm import LLMService


class UIAgent:
    """User Interface Agent for result presentation and user profile update"""
    
    def __init__(self):
        self.user_profile_service = UserProfileService()
        self.llm_service = LLMService()
    
    async def format_recommendations(
        self,
        papers: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        format_type: str = "detailed"
    ) -> Dict[str, Any]:
        """
        格式化推荐结果
        
        Args:
            papers: 排序后的论文列表
            user_profile: 用户画像
            format_type: 格式类型 ('detailed', 'summary', 'json')
            
        Returns:
            格式化后的推荐结果
        """
        if format_type == "detailed":
            formatted_papers = []
            for i, paper in enumerate(papers, 1):
                formatted_papers.append({
                    "rank": i,
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors", []),
                    "abstract": paper.get("abstract", ""),
                    "venue": paper.get("venue"),
                    "year": paper.get("year"),
                    "citation_count": paper.get("citation_count", 0),
                    "url": paper.get("url", ""),
                    "scores": {
                        "final_score": paper.get("final_score", 0),
                        "model_score": paper.get("model_score", 0),
                        "llm_rerank_score": paper.get("llm_rerank_score", 0),
                        "recall_score": paper.get("recall_score", 0)
                    },
                    "recall_source": paper.get("recall_source", "unknown")
                })
            
            return {
                "total": len(formatted_papers),
                "papers": formatted_papers,
                "user_profile": user_profile
            }
        
        elif format_type == "summary":
            # 使用LLM生成推荐摘要
            papers_summary = "\n".join([
                f"{i+1}. {p.get('title', 'N/A')} ({p.get('venue', 'N/A')}, {p.get('year', 'N/A')})"
                for i, p in enumerate(papers[:10])
            ])
            
            prompt = f"""根据以下推荐论文，生成一段简洁的推荐理由（2-3句话）：

{papers_summary}

用户兴趣: {user_profile.get('interests', [])}

请用中文生成推荐理由。"""
            
            try:
                summary = await self.llm_service.generate(prompt)
                return {
                    "summary": summary,
                    "papers": papers[:10],
                    "total": len(papers)
                }
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                return {
                    "summary": "为您推荐了以下论文",
                    "papers": papers[:10],
                    "total": len(papers)
                }
        
        else:  # json
            return {
                "papers": papers,
                "metadata": {
                    "total": len(papers),
                    "user_id": user_profile.get("user_id")
                }
            }
    
    async def update_user_profile(
        self,
        user_id: str,
        recommended_papers: List[Dict[str, Any]],
        user_feedback: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        更新用户画像
        
        Args:
            user_id: 用户ID
            recommended_papers: 推荐的论文列表
            user_feedback: 用户反馈 {'liked': [...], 'disliked': [...], 'saved': [...]}
            
        Returns:
            更新后的用户画像
        """
        try:
            # 获取当前用户画像
            current_profile = await self.user_profile_service.get_profile(user_id)
            
            # 记录曝光历史
            exposed_paper_ids = [p.get("paper_id") for p in recommended_papers]
            await self.user_profile_service.add_exposure_history(
                user_id=user_id,
                paper_ids=exposed_paper_ids
            )
            
            # 如果有用户反馈，更新兴趣
            if user_feedback:
                liked_papers = user_feedback.get("liked", [])
                if liked_papers:
                    # 从喜欢的论文中提取兴趣
                    liked_paper_ids = [p.get("paper_id") if isinstance(p, dict) else p for p in liked_papers]
                    await self.user_profile_service.update_interests_from_papers(
                        user_id=user_id,
                        paper_ids=liked_paper_ids
                    )
                
                # 记录阅读历史
                read_papers = user_feedback.get("read", [])
                if read_papers:
                    read_paper_ids = [p.get("paper_id") if isinstance(p, dict) else p for p in read_papers]
                    await self.user_profile_service.add_read_history(
                        user_id=user_id,
                        paper_ids=read_paper_ids
                    )
            
            # 使用LLM生成/更新用户画像摘要
            updated_profile = await self.user_profile_service.generate_profile_summary(user_id)
            
            logger.info(f"User profile updated for user {user_id}")
            return updated_profile
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            return current_profile if 'current_profile' in locals() else {}
