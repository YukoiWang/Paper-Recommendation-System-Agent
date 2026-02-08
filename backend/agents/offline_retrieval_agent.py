"""
Offline Retrieval Agent - 离线检索
从向量数据库和元数据数据库检索论文
"""
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from backend.services.vector_db import VectorDBService
from backend.services.metadata_db import MetadataDBService
from backend.services.embedding import EmbeddingService


class OfflineRetrievalAgent:
    """Offline Retrieval Agent for RAG-based paper retrieval"""
    
    def __init__(self):
        self.vector_db = VectorDBService()
        self.metadata_db = MetadataDBService()
        self.embedding_service = EmbeddingService()
    
    async def retrieve_by_query(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        根据查询向量检索论文
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            filters: 元数据过滤条件
            
        Returns:
            论文列表（包含向量相似度分数）
        """
        try:
            # 1. 将查询向量化
            query_embedding = await self.embedding_service.embed_query(query)
            
            # 2. 向量相似度检索
            vector_results = await self.vector_db.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # 3. 从元数据库获取完整信息
            paper_ids = [r["paper_id"] for r in vector_results]
            papers = await self.metadata_db.get_papers_by_ids(paper_ids)
            
            # 4. 合并向量相似度分数
            score_map = {r["paper_id"]: r["score"] for r in vector_results}
            for paper in papers:
                paper["vector_score"] = score_map.get(paper["paper_id"], 0.0)
            
            logger.info(f"Retrieved {len(papers)} papers from vector DB")
            return papers
            
        except Exception as e:
            logger.error(f"Offline retrieval error: {e}")
            return []
    
    async def retrieve_by_user_profile(
        self,
        user_profile: Dict[str, Any],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        根据用户画像检索论文
        
        Args:
            user_profile: 用户画像（包含兴趣向量或文本描述）
            top_k: 返回top-k结果
            
        Returns:
            论文列表
        """
        # 从用户画像构建查询
        interests = user_profile.get("interests", [])
        query_text = " ".join(interests) if isinstance(interests, list) else str(interests)
        
        return await self.retrieve_by_query(query_text, top_k=top_k)
    
    async def retrieve_by_paper_id(
        self,
        paper_id: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        基于ItemCF的协同过滤检索（找到相似论文）
        
        Args:
            paper_id: 参考论文ID
            top_k: 返回top-k相似论文
            
        Returns:
            相似论文列表
        """
        try:
            # 1. 获取参考论文的向量
            reference_paper = await self.metadata_db.get_paper_by_id(paper_id)
            if not reference_paper:
                return []
            
            # 2. 使用论文的embedding进行相似度检索
            paper_embedding = await self.vector_db.get_embedding(paper_id)
            if paper_embedding is None:
                return []
            
            # 3. 向量相似度检索
            similar_papers = await self.vector_db.similarity_search(
                query_embedding=paper_embedding,
                top_k=top_k + 1,  # +1 to exclude the reference paper itself
                exclude_ids=[paper_id]
            )
            
            paper_ids = [r["paper_id"] for r in similar_papers]
            papers = await self.metadata_db.get_papers_by_ids(paper_ids)
            
            logger.info(f"Retrieved {len(papers)} similar papers for {paper_id}")
            return papers
            
        except Exception as e:
            logger.error(f"ItemCF retrieval error: {e}")
            return []
