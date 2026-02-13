"""Multi-path recall: vector, rule-based (author/category), ItemCF; then merge by score sum."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from models import Paper, UserProfile

logger = logging.getLogger(__name__)


def vector_recall(user_vector: np.ndarray, store: Any,
                  top_k: int = 50, threshold: float = -1.0,
                  exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
    results = store.search(user_vector, top_k=top_k, exclude_ids=exclude_ids)
    if threshold > -1.0:
        results = [(pid, s) for pid, s in results if s >= threshold]
    logger.info("VectorRecall: %s papers (top_k=%s)", len(results), top_k)
    return results


def rule_based_recall(user: UserProfile, papers: List[Paper],
                      author_boost: float = 0.25,
                      category_boost: float = 0.10) -> List[Tuple[str, float]]:
    if not user.followed_authors and not user.preferred_categories:
        return []
    followed = {a.lower().strip() for a in user.followed_authors}
    preferred = {c.lower().strip() for c in user.preferred_categories}
    results = []
    for p in papers:
        bonus = 0.0
        if followed and {a.lower().strip() for a in p.authors} & followed:
            bonus += author_boost
        if preferred and {c.lower().strip() for c in p.categories} & preferred:
            bonus += category_boost
        if bonus > 0:
            results.append((p.paper_id, bonus))
    logger.info("RuleRecall: %s papers", len(results))
    return results


def itemcf_recall(user: UserProfile, store: Any,
                  paper_cache: Dict[str, Paper],
                  top_k_per_item: int = 5,
                  top_k_total: int = 20) -> List[Tuple[str, float]]:
    if not user.liked_items:
        return []
    exclude = set(user.liked_items)
    score_map: Dict[str, float] = {}
    for liked_id in user.liked_items:
        paper = paper_cache.get(liked_id)
        vec = paper.embedding if (paper is not None and getattr(paper, "embedding", None) is not None) else store.get_vector(liked_id)
        if vec is None:
            continue
        neighbors = store.search(vec, top_k=top_k_per_item + len(exclude), exclude_ids=exclude)
        for pid, sc in neighbors[:top_k_per_item]:
            score_map[pid] = max(score_map.get(pid, 0.0), sc)
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    results = sorted_items[:top_k_total]
    logger.info("ItemCF: %s liked -> %s candidates", len(user.liked_items), len(results))
    return results


def merge_results(*recall_lists: List[Tuple[str, float]],
                  top_k: int = 30) -> List[Tuple[str, float]]:
    merged: Dict[str, float] = {}
    for rlist in recall_lists:
        for pid, sc in rlist:
            merged[pid] = merged.get(pid, 0.0) + sc
    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    final = sorted_items[:top_k]
    logger.info("Merge: %s input -> %s unique -> %s final",
                sum(len(r) for r in recall_lists), len(merged), len(final))
    return final
