"""Optional Semantic Scholar search (design §5 search_s2)."""
from __future__ import annotations

import logging
import os
from typing import List

from agent.models import Paper

logger = logging.getLogger(__name__)


def search_s2(query: str, k: int = 8) -> List[Paper]:
    """Best-effort S2 lookup. Returns [] when no API key or the request fails."""
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    if not key or not (query or "").strip():
        return []
    try:
        import httpx
        headers = {"x-api-key": key}
        params = {
            "query": query.strip()[:200],
            "limit": k,
            "fields": "paperId,title,abstract,year,authors,externalIds",
        }
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, headers=headers, timeout=8.0,
        )
        if r.status_code != 200:
            return []
        out: List[Paper] = []
        for item in (r.json() or {}).get("data") or []:
            pid = str(item.get("paperId") or "")
            title = item.get("title") or ""
            if not pid or not title:
                continue
            authors = [a.get("name") for a in (item.get("authors") or []) if a.get("name")]
            year = str(item.get("year") or "")
            out.append(Paper(
                paper_id=pid, title=title,
                abstract=item.get("abstract") or "",
                authors=authors, published=year,
            ))
        return out
    except Exception as e:
        logger.warning("search_s2 failed: %s", e)
        return []
