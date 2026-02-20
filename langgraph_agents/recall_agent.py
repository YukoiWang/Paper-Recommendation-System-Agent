"""Recall agent: multi-path (vector + rule + ItemCF) merge + online/offline fusion."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent.models import Paper, UserProfile, RecommendationResult
from agent.cold_start import resolve_user_vector, get_trending_papers
from agent.recall_strategies import vector_recall, rule_based_recall, itemcf_recall, merge_results

logger = logging.getLogger(__name__)


def _merge_online_offline(
    online: List[Paper],
    offline: List[Paper],
    ratio: float,
) -> List[Paper]:
    """Fuse online and offline results. ratio=1 -> all online first, ratio=0 -> all offline first."""
    seen = set()
    fused = []
    n_online = max(1, int(len(online) * ratio)) if online else 0
    n_offline = max(1, int(len(offline) * (1 - ratio))) if offline else 0
    for p in online[:n_online]:
        if p.paper_id not in seen:
            fused.append(p)
            seen.add(p.paper_id)
    for p in offline[:n_offline]:
        if p.paper_id not in seen:
            fused.append(p)
            seen.add(p.paper_id)
    if len(fused) < len(online) + len(offline):
        for p in online[n_online:] + offline[n_offline:]:
            if p.paper_id not in seen:
                fused.append(p)
                seen.add(p.paper_id)
    return fused


def _apply_recency_boost(papers: List[Paper], recency_weight: float = 0.3) -> List[Paper]:
    """Sort by score * (1 - recency_weight) + recency_score * recency_weight."""
    import re
    def parse_year(pub: str) -> int:
        if not pub:
            return 0
        m = re.search(r"20\d{2}", str(pub))
        return int(m.group()) if m else 0
    max_year = max(parse_year(p.published) for p in papers) or 2024
    scored = []
    for p in papers:
        yr = parse_year(p.published)
        recency = max(0, (yr - 2020) / 4.0) if max_year > 2020 else 0
        norm_recency = recency / 4.0 if max_year > 2020 else 0
        combined = p.score * (1 - recency_weight) + norm_recency * recency_weight
        scored.append((combined, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


class RecallAgent:
    """
    Multi-path recall: vector + rule + ItemCF, then merge.
    Also fuses online search results with offline recall.
    """

    def __init__(
        self,
        retrieval_agent,
        top_k_vector: int = 50,
        top_k_itemcf: int = 20,
        top_k_final: int = 30,
        author_boost: float = 0.25,
        category_boost: float = 0.10,
        trending_count: int = 10,
    ):
        self.retrieval = retrieval_agent
        self.top_k_vector = top_k_vector
        self.top_k_itemcf = top_k_itemcf
        self.top_k_final = top_k_final
        self.author_boost = author_boost
        self.category_boost = category_boost
        self.trending_count = trending_count

    def offline_recall(
        self,
        user: UserProfile,
        new_papers: Optional[List[Paper]] = None,
        top_k_override: Optional[int] = None,
        prefer_recent: bool = False,
    ) -> RecommendationResult:
        """Offline multi-path recall: vector + rule + ItemCF."""
        all_papers = list(new_papers) if new_papers else list(self.retrieval._cache.values())
        if new_papers:
            self.retrieval.index_papers(new_papers)
        if self.retrieval.store.size == 0:
            logger.warning("Vector store empty")
            return RecommendationResult(user_id=user.user_id, recall_stats={"error": "empty"})
        top_k = top_k_override if top_k_override is not None else self.top_k_final
        user_vec, is_cold = resolve_user_vector(user, self.retrieval.embedder, all_papers)
        stats = {}
        vec_results = vector_recall(user_vec, self.retrieval.store, top_k=self.top_k_vector)
        stats["vector_recall"] = len(vec_results)
        rule_results = rule_based_recall(
            user, all_papers,
            author_boost=self.author_boost,
            category_boost=self.category_boost,
        )
        stats["rule_recall"] = len(rule_results)
        icf_results = itemcf_recall(user, self.retrieval.store, self.retrieval._cache, top_k_total=self.top_k_itemcf)
        stats["itemcf_recall"] = len(icf_results)
        merged = merge_results(vec_results, rule_results, icf_results, top_k=top_k)
        stats["merged"] = len(merged)
        if is_cold and len(merged) < self.trending_count:
            existing_ids = {pid for pid, _ in merged}
            for tp in get_trending_papers(all_papers, self.trending_count):
                if tp.paper_id not in existing_ids:
                    merged.append((tp.paper_id, 0.0))
                    existing_ids.add(tp.paper_id)
                if len(merged) >= top_k:
                    break
            stats["trending_supplement"] = len(merged) - stats["merged"]
        recommended = []
        for pid, score in merged:
            paper = self.retrieval._cache.get(pid)
            if paper:
                rec = Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories,
                    published=paper.published, embedding=paper.embedding, score=score,
                )
                recommended.append(rec)
        if prefer_recent and recommended:
            recommended = _apply_recency_boost(recommended)
            recommended = recommended[:top_k]
        result = RecommendationResult(user_id=user.user_id, recommended_papers=recommended, recall_stats=stats)
        logger.info("Stats: %s | final %s papers", stats, len(recommended))
        return result

    def fuse(
        self,
        online: List[Paper],
        offline: List[Paper],
        ratio: float = 0.5,
    ) -> List[Paper]:
        """Merge online and offline recall results."""
        return _merge_online_offline(online, offline, ratio)
