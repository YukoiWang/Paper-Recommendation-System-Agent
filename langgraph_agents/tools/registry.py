"""Shared tools wrapping existing retrieval / recall / rank / arxiv (design §5)."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from agent.models import Paper, UserProfile
from langgraph_agents.online_search_agent import fetch_arxiv_papers, _extract_query_keywords
from langgraph_agents.tools.gather_evidence import gather_evidence as _gather_evidence
from langgraph_agents.tools.read_fulltext import read_fulltext as _read_fulltext
from langgraph_agents.tools.search_s2 import search_s2 as _search_s2

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, retrieval=None, recall=None, rank=None):
        self.retrieval = retrieval
        self.recall = recall
        self.rank = rank

    def search_local(self, query: str, k: int = 10, hyde_document: str = "") -> List[Paper]:
        if self.retrieval is None or not (query or "").strip():
            return []
        return self.retrieval.retrieve_by_query(
            query.strip(), top_k=k, hyde_document=hyde_document or None,
        )

    def search_arxiv(
        self,
        query: str,
        k: int = 10,
        categories: Optional[List[str]] = None,
        days: int = 365,
    ) -> List[Paper]:
        keywords = _extract_query_keywords(query) or [w for w in query.split() if len(w) > 1][:6]
        if not keywords:
            keywords = ["machine learning"]
        cats = categories or ["cs.LG", "cs.AI", "cs.CL"]
        return fetch_arxiv_papers(keywords=keywords, categories=cats, days=days, max_results=k)

    def search_s2(self, query: str, k: int = 8) -> List[Paper]:
        return _search_s2(query, k=k)

    def recall_multi(
        self,
        profile: UserProfile,
        query: Optional[str] = None,
        online_papers: Optional[List[Paper]] = None,
        top_k: int = 20,
        prefer_recent: bool = False,
        use_itemcf: bool = True,
    ) -> List[Paper]:
        if self.recall is None:
            q = (query or (profile.interest_text if profile else "") or "").strip()
            return self.search_local(q, k=top_k) if q else []
        if not use_itemcf:
            q = (query or (profile.interest_text if profile else "") or "").strip()
            return self.search_local(q, k=top_k) if q else []
        rr = self.recall.offline_recall(
            user=profile,
            new_papers=online_papers,
            top_k_override=top_k,
            prefer_recent=prefer_recent,
        )
        return list(rr.recommended_papers or [])

    def rerank(
        self,
        papers: Sequence[Paper],
        query: str,
        user: Optional[UserProfile] = None,
        top_k: int = 10,
    ) -> List[Paper]:
        papers = list(papers)
        if not papers:
            return []
        if self.rank is None:
            return papers[:top_k]
        profile = user or UserProfile(user_id="anonymous")
        return self.rank.rerank(papers, query=query or "", user=profile, top_k=top_k)

    def gather_evidence(self, papers: Sequence[Paper], question: str, max_chunks: int = 12) -> List[dict]:
        return _gather_evidence(papers, question, max_chunks=max_chunks, fulltext_fn=self.read_fulltext)

    def read_fulltext(self, paper_id: str) -> str:
        return _read_fulltext(paper_id)
