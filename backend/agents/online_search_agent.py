"""
Online Search Agent - 在线搜索新论文
从arXiv、Semantic Scholar等API获取最新论文
"""
from typing import List, Dict, Any
import httpx
from loguru import logger

from backend.config import settings


class OnlineSearchAgent:
    """Online Search Agent for fetching papers from APIs"""
    
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        self.semantic_scholar_api_key = settings.SEMANTIC_SCHOLAR_API_KEY
    
    async def search_arxiv(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        从arXiv搜索论文
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            sort_by: 排序字段
            sort_order: 排序顺序
            
        Returns:
            论文列表
        """
        try:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(self.arxiv_base_url, params=params)
                response.raise_for_status()
                
                # TODO: Parse XML response and convert to structured format
                papers = []
                logger.info(f"ArXiv search returned {len(papers)} papers")
                return papers
                
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []
    
    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 50,
        fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        从Semantic Scholar搜索论文
        
        Args:
            query: 搜索查询
            limit: 结果限制
            fields: 返回字段列表
            
        Returns:
            论文列表
        """
        if fields is None:
            fields = [
                "paperId", "title", "authors", "abstract", "venue",
                "year", "citationCount", "influentialCitationCount"
            ]
        
        try:
            url = f"{self.semantic_scholar_base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": ",".join(fields)
            }
            
            headers = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                papers = data.get("data", [])
                logger.info(f"Semantic Scholar search returned {len(papers)} papers")
                return papers
                
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        综合搜索多个来源
        
        Args:
            query: 搜索查询
            sources: 搜索来源列表 ['arxiv', 'semantic_scholar']
            max_results: 每个来源的最大结果数
            
        Returns:
            合并后的论文列表
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]
        
        all_papers = []
        
        if "arxiv" in sources:
            arxiv_papers = await self.search_arxiv(query, max_results=max_results)
            all_papers.extend(arxiv_papers)
        
        if "semantic_scholar" in sources:
            ss_papers = await self.search_semantic_scholar(query, limit=max_results)
            all_papers.extend(ss_papers)
        
        # TODO: Deduplicate papers by title/paperId
        logger.info(f"Online search returned {len(all_papers)} total papers")
        return all_papers
