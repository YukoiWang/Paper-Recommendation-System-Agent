"""
Recall Agent - 召回候选集合
整合Online Search和Offline Retrieval的结果，生成候选集
"""
from typing import List, Dict, Any
from loguru import logger

from backend.agents.online_search_agent import OnlineSearchAgent
from backend.agents.offline_retrieval_agent import OfflineRetrievalAgent


class RecallAgent:
    """Recall Agent for merging candidates from multiple sources"""
    
    def __init__(self):
        self.online_search = OnlineSearchAgent()
        self.offline_retrieval = OfflineRetrievalAgent()
    
    async def recall(
        self,
        user_query: str,
        user_profile: Dict[str, Any],
        recall_strategies: List[str] = None,
        max_candidates: int = 500
    ) -> List[Dict[str, Any]]:
        """
        多路召回生成候选集
        
        Args:
            user_query: 用户查询
            user_profile: 用户画像
            recall_strategies: 召回策略列表 ['online_search', 'rag', 'itemcf']
            max_candidates: 最大候选数
            
        Returns:
            候选论文列表
        """
        if recall_strategies is None:
            recall_strategies = ["online_search", "rag", "itemcf"]
        
        all_candidates = []
        
        # 1. Online Search召回
        if "online_search" in recall_strategies:
            try:
                online_papers = await self.online_search.search(
                    query=user_query,
                    max_results=max_candidates // len(recall_strategies)
                )
                for paper in online_papers:
                    paper["recall_source"] = "online_search"
                    paper["recall_score"] = 1.0  # 在线搜索默认分数
                all_candidates.extend(online_papers)
                logger.info(f"Online search recalled {len(online_papers)} papers")
            except Exception as e:
                logger.error(f"Online search recall error: {e}")
        
        # 2. RAG召回（向量检索）
        if "rag" in recall_strategies:
            try:
                rag_papers = await self.offline_retrieval.retrieve_by_user_profile(
                    user_profile=user_profile,
                    top_k=max_candidates // len(recall_strategies)
                )
                for paper in rag_papers:
                    paper["recall_source"] = "rag"
                    paper["recall_score"] = paper.get("vector_score", 0.0)
                all_candidates.extend(rag_papers)
                logger.info(f"RAG recalled {len(rag_papers)} papers")
            except Exception as e:
                logger.error(f"RAG recall error: {e}")
        
        # 3. ItemCF召回（协同过滤）
        if "itemcf" in recall_strategies:
            try:
                # 从用户历史中获取最近阅读的论文
                recent_papers = user_profile.get("recent_reads", [])
                if recent_papers:
                    # 对每个最近阅读的论文找相似论文
                    itemcf_papers = []
                    for paper_id in recent_papers[:5]:  # 最多取5篇最近阅读的论文
                        similar = await self.offline_retrieval.retrieve_by_paper_id(
                            paper_id=paper_id,
                            top_k=max_candidates // (len(recall_strategies) * 5)
                        )
                        for paper in similar:
                            paper["recall_source"] = "itemcf"
                            paper["recall_score"] = paper.get("vector_score", 0.0)
                        itemcf_papers.extend(similar)
                    
                    all_candidates.extend(itemcf_papers)
                    logger.info(f"ItemCF recalled {len(itemcf_papers)} papers")
            except Exception as e:
                logger.error(f"ItemCF recall error: {e}")
        
        # 4. 去重（基于paper_id或title）
        candidates = self._deduplicate(all_candidates)
        
        # 5. 限制候选数量
        candidates = candidates[:max_candidates]
        
        logger.info(f"Total recalled {len(candidates)} unique candidates")
        return candidates
    
    def _deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重候选论文
        
        Args:
            papers: 论文列表
            
        Returns:
            去重后的论文列表
        """
        seen_ids = set()
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            paper_id = paper.get("paper_id") or paper.get("id")
            title = paper.get("title", "").lower().strip()
            
            # 使用paper_id或title去重
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
            elif title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
