"""Online search agent: ArXiv + Semantic Scholar. Merged from agent/online_agent and backend/online_search_agent."""
from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.models import Paper, UserProfile

logger = logging.getLogger(__name__)


def _profile_to_keywords(profile: Optional[UserProfile]) -> Tuple[List[str], List[str]]:
    """Extract keywords and categories from UserProfile."""
    if profile is None:
        return [], []
    keywords = []
    if profile.interest_text:
        keywords.extend(profile.interest_text.split())
    if profile.special_requirements:
        keywords.extend(profile.special_requirements)
    keywords = list(dict.fromkeys(kw for kw in keywords if len(kw) > 1))[:5]
    categories = list(profile.preferred_categories or []) or ["cs.LG", "cs.AI", "cs.CL"]
    return keywords or ["machine learning"], categories


def fetch_arxiv_papers(
    keywords: List[str],
    categories: List[str],
    days: int = 30,
    max_results: int = 20,
) -> List[Paper]:
    """
    Fetch papers from ArXiv API (arxiv package). Returns Paper objects.
    From agent/online_agent.
    """
    try:
        import arxiv
    except ImportError as e:
        logger.warning("arxiv package not installed: %s", e)
        return []
    now = datetime.now()
    past_date = now - timedelta(days=days)
    start_str = past_date.strftime("%Y%m%d%H%M")
    end_str = now.strftime("%Y%m%d%H%M")
    keyword_part = " OR ".join([f'"{k}"' for k in keywords])
    category_part = " OR ".join([f"cat:{c}" for c in categories])
    final_query = f"({keyword_part}) AND ({category_part}) AND submittedDate:[{start_str} TO {end_str}]"
    try:
        search = arxiv.Search(
            query=final_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        client = arxiv.Client()
        results = []
        for r in client.results(search):
            clean_title = r.title.replace("\n", " ")
            clean_summary = r.summary.replace("\n", " ")
            results.append(Paper(
                paper_id=r.get_short_id(),
                title=clean_title,
                abstract=clean_summary,
                authors=[a.name for a in r.authors],
                published=str(r.published.date()),
                categories=r.categories,
            ))
        logger.info("ArXiv fetch: %s papers (keywords=%s)", len(results), keywords[:3])
        return results
    except Exception as e:
        logger.warning("ArXiv fetch failed: %s", e)
        return []


async def fetch_semantic_scholar_papers(
    query: str,
    limit: int = 50,
    api_key: Optional[str] = None,
) -> List[Paper]:
    """
    Fetch papers from Semantic Scholar API. Returns Paper objects.
    From backend/online_search_agent, converted to Paper.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; Semantic Scholar unavailable")
        return []
    api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "paperId,title,authors,abstract,year,venue"}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
        papers = []
        for item in data.get("data", []):
            auth = item.get("authors", [])
            authors = [a.get("name", "") for a in auth] if isinstance(auth, list) else []
            paper_id = item.get("paperId", "") or item.get("title", "")[:20]
            papers.append(Paper(
                paper_id=paper_id,
                title=(item.get("title") or "").replace("\n", " "),
                abstract=(item.get("abstract") or "").replace("\n", " "),
                authors=authors,
                published=str(item.get("year", "")),
                categories=[],
            ))
        logger.info("Semantic Scholar fetch: %s papers", len(papers))
        return papers
    except Exception as e:
        logger.warning("Semantic Scholar fetch failed: %s", e)
        return []


class OnlineSearchAgent:
    """
    Fetch papers from ArXiv and optionally Semantic Scholar.
    Merged from agent/online_agent and backend/online_search_agent.
    """

    def __init__(
        self,
        time_window_days: int = 30,
        max_results: int = 20,
        sources: Optional[List[str]] = None,
        semantic_scholar_api_key: Optional[str] = None,
    ):
        self.time_window_days = time_window_days
        self.max_results = max_results
        self.sources = sources or ["arxiv"]
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    def run(self, state: dict) -> dict:
        """Sync: read state, fetch papers, update state.online_search_result."""
        profile = state.get("user_profile")
        query = state.get("user_query", "")
        keywords, categories = _profile_to_keywords(profile)
        if query and not keywords:
            keywords = [w for w in query.split() if len(w) > 2][:5] or ["machine learning"]
        if not keywords:
            keywords = ["machine learning"]
        all_papers = []
        if "arxiv" in self.sources:
            papers = fetch_arxiv_papers(
                keywords=keywords,
                categories=categories,
                days=self.time_window_days,
                max_results=self.max_results,
            )
            all_papers.extend(papers)
        state["online_search_result"] = all_papers
        logger.info("OnlineSearchAgent: wrote %s papers to state", len(all_papers))
        return state

    async def run_async(self, state: dict) -> dict:
        """Async: same as run but can call Semantic Scholar."""
        profile = state.get("user_profile")
        query = state.get("user_query", "")
        keywords, categories = _profile_to_keywords(profile)
        if query and not keywords:
            keywords = [w for w in query.split() if len(w) > 2][:5] or ["machine learning"]
        if not keywords:
            keywords = ["machine learning"]
        query_str = " ".join(keywords)
        all_papers = []
        if "arxiv" in self.sources:
            papers = fetch_arxiv_papers(
                keywords=keywords,
                categories=categories,
                days=self.time_window_days,
                max_results=self.max_results,
            )
            all_papers.extend(papers)
        if "semantic_scholar" in self.sources:
            ss_papers = await fetch_semantic_scholar_papers(
                query=query_str,
                limit=self.max_results,
                api_key=self.semantic_scholar_api_key,
            )
            seen_ids = {p.paper_id for p in all_papers}
            for p in ss_papers:
                if p.paper_id not in seen_ids:
                    all_papers.append(p)
                    seen_ids.add(p.paper_id)
        state["online_search_result"] = all_papers
        logger.info("OnlineSearchAgent: wrote %s papers to state", len(all_papers))
        return state
