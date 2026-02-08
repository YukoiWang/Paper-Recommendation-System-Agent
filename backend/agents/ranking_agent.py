"""
Ranking Agent - 排序
双阶段排序：传统模型排序 + LLM语义重排
"""
from typing import List, Dict, Any
import numpy as np
from loguru import logger

from backend.services.ranking_model import RankingModelService
from backend.services.llm import LLMService


class RankingAgent:
    """Ranking Agent for two-stage ranking"""
    
    def __init__(self):
        self.ranking_model = RankingModelService()
        self.llm_service = LLMService()
    
    async def rank(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        top_n: int = 50,
        use_llm_rerank: bool = True,
        llm_rerank_top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        双阶段排序
        
        Args:
            candidates: 候选论文列表
            user_profile: 用户画像
            top_n: 最终返回的top-n
            use_llm_rerank: 是否使用LLM重排
            llm_rerank_top_k: LLM重排的候选数
            
        Returns:
            排序后的论文列表
        """
        if not candidates:
            return []
        
        # Stage 1: 传统模型排序（LightGBM/DeepFM）
        try:
            ranked_candidates = await self.ranking_model.rank(
                candidates=candidates,
                user_profile=user_profile
            )
            logger.info(f"Model ranking completed, top score: {ranked_candidates[0].get('model_score', 0) if ranked_candidates else 0}")
        except Exception as e:
            logger.error(f"Model ranking error: {e}, using recall scores")
            # Fallback: 使用recall_score排序
            ranked_candidates = sorted(
                candidates,
                key=lambda x: x.get("recall_score", 0.0),
                reverse=True
            )
        
        # Stage 2: LLM语义重排（可选）
        if use_llm_rerank and len(ranked_candidates) > llm_rerank_top_k:
            try:
                # 取top-k进行LLM重排
                top_k_candidates = ranked_candidates[:llm_rerank_top_k * 2]  # 取更多候选供LLM选择
                reranked = await self.llm_rerank(
                    candidates=top_k_candidates,
                    user_profile=user_profile,
                    top_k=llm_rerank_top_k
                )
                
                # 合并：LLM重排的top-k + 剩余的模型排序结果
                reranked_ids = {p["paper_id"] for p in reranked}
                remaining = [p for p in ranked_candidates if p["paper_id"] not in reranked_ids]
                
                final_ranked = reranked + remaining
                logger.info(f"LLM reranking completed, top {len(reranked)} papers reranked")
            except Exception as e:
                logger.error(f"LLM reranking error: {e}, using model ranking results")
                final_ranked = ranked_candidates
        else:
            final_ranked = ranked_candidates
        
        # 返回top-n
        return final_ranked[:top_n]
    
    async def llm_rerank(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        LLM语义重排
        
        Args:
            candidates: 候选论文列表
            user_profile: 用户画像
            top_k: 返回top-k
            
        Returns:
            LLM重排后的论文列表
        """
        # 构建prompt
        user_interests = user_profile.get("interests", [])
        user_interests_str = ", ".join(user_interests) if isinstance(user_interests, list) else str(user_interests)
        
        papers_info = []
        for i, paper in enumerate(candidates):
            papers_info.append(
                f"{i+1}. Title: {paper.get('title', 'N/A')}\n"
                f"   Abstract: {paper.get('abstract', 'N/A')[:200]}...\n"
                f"   Authors: {', '.join(paper.get('authors', [])[:3])}\n"
                f"   Venue: {paper.get('venue', 'N/A')}\n"
            )
        
        prompt = f"""你是一个学术论文推荐专家。根据用户的兴趣和研究方向，对以下论文进行重新排序。

用户兴趣: {user_interests_str}

候选论文:
{''.join(papers_info)}

请根据论文与用户兴趣的相关性、论文质量、新颖性等因素，返回排序后的论文编号（用逗号分隔，例如：3,1,5,2,4...）。
只返回编号，不要其他内容。"""

        try:
            response = await self.llm_service.generate(prompt)
            # 解析响应，提取排序后的编号
            ranked_indices = self._parse_llm_ranking(response, len(candidates))
            
            # 根据LLM的排序重新排列
            reranked = [candidates[i] for i in ranked_indices if 0 <= i < len(candidates)]
            
            # 添加LLM重排分数
            for i, paper in enumerate(reranked):
                paper["llm_rerank_score"] = len(reranked) - i
                paper["final_score"] = (
                    paper.get("model_score", 0) * 0.7 +
                    paper.get("llm_rerank_score", 0) * 0.3
                )
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"LLM rerank parsing error: {e}")
            # Fallback: 返回原始顺序
            return candidates[:top_k]
    
    def _parse_llm_ranking(self, response: str, max_index: int) -> List[int]:
        """
        解析LLM返回的排序结果
        
        Args:
            response: LLM响应文本
            max_index: 最大索引值
            
        Returns:
            排序后的索引列表
        """
        try:
            # 提取数字
            import re
            numbers = re.findall(r'\d+', response)
            indices = [int(n) - 1 for n in numbers if n.isdigit()]  # 转换为0-based索引
            
            # 过滤有效索引
            valid_indices = [i for i in indices if 0 <= i < max_index]
            
            # 如果解析失败，返回原始顺序
            if not valid_indices:
                return list(range(min(max_index, 20)))
            
            return valid_indices
        except Exception as e:
            logger.error(f"Failed to parse LLM ranking: {e}")
            return list(range(min(max_index, 20)))
