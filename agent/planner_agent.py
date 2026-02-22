"""Planner agent: orchestrates QA, retrieval, rank, online search based on blackboard state."""
from __future__ import annotations
import logging
import uuid
from typing import Any, Dict, List, Optional

from blackboard import Blackboard
from agent.models import Paper, UserProfile, is_profile_sufficient
from qa_agent import Intent, classify_intent

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


class PlannerAgent:
    """
    Central orchestrator. Reads blackboard, decides which agents to call,
    writes results back to blackboard.
    """

    def __init__(
        self,
        retrieval_agent,
        rank_agent,
        qa_agent,
        online_agent=None,
        default_top_k: int = 10,
    ):
        self.retrieval = retrieval_agent
        self.rank = rank_agent
        self.qa = qa_agent
        self.online = online_agent
        self.default_top_k = default_top_k

    def run_daily_rec(self, blackboard: Blackboard) -> str:
        """
        Daily recommendation flow.
        - If profile sufficient: online + recall + fuse + rank -> QA push
        - Else: QA asks for interests -> return
        """
        bb = blackboard
        bb.clear_turn_state()
        bb.is_daily_rec = True
        bb.top_k = bb.top_k or self.default_top_k
        profile = bb.user_profile

        if not is_profile_sufficient(profile):
            bb.needs_profile_clarification = True
            status = self.qa.process_turn(bb)
            return status

        if self.online:
            self.online.run(bb)
        rr = self.retrieval.retrieve_for_user(
            profile,
            new_papers=bb.online_search_result if bb.online_search_result else None,
            top_k_override=bb.top_k * 2,
            prefer_recent=bb.prefer_latest_papers,
        )
        bb.recall_result = list(rr.recommended_papers)
        bb.recall_stats = dict(rr.recall_stats)
        fused = _merge_online_offline(
            bb.online_search_result,
            bb.recall_result,
            bb.online_offline_fusion_ratio,
        )
        bb.fused_candidates = fused
        ranked = self.rank.rerank(
            fused,
            query=bb.user_query or profile.interest_text or "",
            user=profile,
            top_k=bb.top_k,
        )
        bb.ranked_papers = ranked
        bb.final_papers = ranked[: bb.top_k]
        bb.qa_intent = Intent.RECOMMEND
        status = self.qa.process_turn(bb)
        return status

    def run_user_query(self, blackboard: Blackboard) -> str:
        """
        User active query. Route by intent from QA.
        - recommend/qa/explain/compare/summarize: retrieval + rank -> QA
        - recommend + profile incomplete: QA asks first
        - recommend + want latest: online + higher fusion + prefer_recent
        - general: if history has papers, QA answers; else retrieval first
        """
        bb = blackboard
        bb.clear_turn_state()
        bb.is_daily_rec = False
        bb.top_k = bb.top_k or self.default_top_k
        query = bb.user_query or ""
        profile = bb.user_profile or UserProfile(user_id=bb.user_id or "anonymous")

        intent = classify_intent(query)
        bb.qa_intent = intent

        if "latest" in query.lower() or "new" in query.lower() or "recent" in query.lower():
            bb.prefer_latest_papers = True
            bb.online_offline_fusion_ratio = 0.7

        if intent == Intent.RECOMMEND:
            if not is_profile_sufficient(profile):
                bb.needs_profile_clarification = True
                return self.qa.process_turn(bb)

            if bb.prefer_latest_papers and self.online:
                self.online.run(bb)

            if bb.online_search_result and bb.prefer_latest_papers:
                self.retrieval.index_papers(bb.online_search_result)

            rr = self.retrieval.retrieve_by_query(
                query=query,
                user=profile,
                top_k=bb.top_k * 2,
                prefer_recent=bb.prefer_latest_papers,
            )
            offline = rr
        elif intent in (Intent.QA, Intent.EXPLAIN, Intent.COMPARE, Intent.SUMMARIZE):
            rr = self.retrieval.retrieve_by_query(
                query=query,
                user=profile,
                top_k=bb.top_k * 2,
                prefer_recent=bb.prefer_latest_papers,
            )
            offline = rr
        else:
            status = self.qa.process_turn(bb)
            if status == "need_retrieval":
                rr = self.retrieval.retrieve_by_query(
                    query=query,
                    user=profile,
                    top_k=bb.top_k * 2,
                )
                bb.recall_result = rr
                fused = _merge_online_offline(
                    bb.online_search_result,
                    rr,
                    bb.online_offline_fusion_ratio,
                )
                bb.fused_candidates = fused
                ranked = self.rank.rerank(
                    fused,
                    query=query,
                    user=profile,
                    top_k=bb.top_k,
                )
                bb.ranked_papers = ranked
                bb.final_papers = ranked[: bb.top_k]
                return self.qa.process_turn(bb)
            return status

        bb.recall_result = offline
        fused = _merge_online_offline(
            bb.online_search_result,
            offline,
            bb.online_offline_fusion_ratio,
        )
        bb.fused_candidates = fused
        ranked = self.rank.rerank(
            fused,
            query=query,
            user=profile,
            top_k=bb.top_k,
        )
        bb.ranked_papers = ranked
        bb.final_papers = ranked[: bb.top_k]
        return self.qa.process_turn(bb)

    def run(
        self,
        blackboard: Blackboard,
        is_daily: bool = False,
        is_feedback: bool = False,
    ) -> str:
        """
        Main entry: run daily rec or user query.
        When is_feedback=True, user_query is treated as feedback on previous recommendation.
        """
        if is_feedback and blackboard.user_query:
            blackboard.user_feedback = blackboard.user_query
            blackboard.user_query = ""
        if is_daily:
            return self.run_daily_rec(blackboard)
        return self.run_user_query(blackboard)
