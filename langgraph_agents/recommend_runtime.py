"""Recommend subgraph: daily / query recommend share recall + rerank (design §7)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from agent.models import Paper, UserProfile
from langgraph_agents.policy import THRESHOLDS
from langgraph_agents.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_recommend(state: Dict[str, Any], tools: ToolRegistry) -> Dict[str, Any]:
    profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
    top_k = int(state.get("top_k") or 10)
    wo = state.get("work_order") or {}
    intent = wo.get("intent") or "recommend"
    topic = (wo.get("topic") or state.get("user_query") or profile.interest_text or "").strip()
    need_sota = bool(wo.get("need_sota") or state.get("prefer_latest_papers"))
    is_daily = intent == "daily" or bool(state.get("is_daily_rec"))
    if intent == "feedback":
        is_daily = False
        topic = topic or profile.interest_text or ""
        need_sota = need_sota or "prefer_recent" in list(profile.special_requirements or [])
    seen: Set[str] = set(state.get("daily_seen_ids") or [])
    min_c = int(THRESHOLDS.get("min_candidates", 3))

    online: List[Paper] = []
    if need_sota:
        online = tools.search_arxiv(topic or "machine learning", k=top_k)
        if online:
            state["recovery"] = "switch_arxiv"

    use_itemcf = is_daily and bool(profile.liked_items or profile.interest_text)
    if is_daily:
        papers = tools.recall_multi(
            profile, query=None, online_papers=online or None,
            top_k=top_k * 2, prefer_recent=need_sota, use_itemcf=use_itemcf,
        )
        rank_q = profile.interest_text or topic
    else:
        local = tools.search_local(topic, k=top_k * 2) if topic else []
        extra = tools.recall_multi(
            profile, query=topic, top_k=max(5, top_k // 2), use_itemcf=False,
        ) if profile and (profile.interest_text or profile.preferred_categories) else []
        papers = _dedupe(local + extra + online)
        rank_q = topic
        if len(local) < min_c and not need_sota and topic:
            more = tools.search_arxiv(topic, k=top_k)
            papers = _dedupe(papers + more)
            if more:
                state["recovery"] = "switch_arxiv"
                state["failure_type"] = "empty_retrieval" if not local else "low_relevance"

    papers = [p for p in papers if getattr(p, "paper_id", None) not in seen]
    if not papers and is_daily:
        state["failure_type"] = "empty_retrieval"

    ranked = tools.rerank(papers, query=rank_q, user=profile, top_k=top_k)
    state["fused_candidates"] = papers
    state["ranked_papers"] = ranked
    state["final_papers"] = ranked[:top_k]
    state["retrieval_result"] = papers
    state["plan_ledger"] = {"goal": rank_q, "intent": intent, "daily": is_daily}
    logger.info("recommend: daily=%s n_rank=%s", is_daily, len(ranked))
    return state


def _dedupe(papers: List[Paper]) -> List[Paper]:
    seen = set()
    out = []
    for p in papers:
        pid = getattr(p, "paper_id", None) or id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out
