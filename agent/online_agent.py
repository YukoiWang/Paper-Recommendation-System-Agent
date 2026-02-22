"""Online search agent: fetch latest papers from ArXiv API by user profile or query."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

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
    Fetch papers from ArXiv API. Returns agent.models.Paper objects.
    Lazy-imports arxiv to avoid hard dependency when offline-only.
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


class OnlineSearchAgent:
    """
    Fetch latest papers from ArXiv. Reads profile/query from blackboard,
    writes online_search_result to blackboard.
    """

    def __init__(self, time_window_days: int = 30, max_results: int = 20):
        self.time_window_days = time_window_days
        self.max_results = max_results

    def run(self, blackboard) -> None:
        """Read blackboard, fetch papers, write to blackboard.online_search_result."""
        profile = blackboard.user_profile
        query = blackboard.user_query or ""
        keywords, categories = _profile_to_keywords(profile)
        if query and not keywords:
            keywords = [w for w in query.split() if len(w) > 2][:5] or ["machine learning"]
        if not keywords:
            keywords = ["machine learning"]
        papers = fetch_arxiv_papers(
            keywords=keywords,
            categories=categories,
            days=self.time_window_days,
            max_results=self.max_results,
        )
        blackboard.online_search_result = papers
        blackboard.online_search_ids = [p.paper_id for p in papers]
        logger.info("OnlineSearchAgent: wrote %s papers to blackboard", len(papers))
