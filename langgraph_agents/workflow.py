"""LangGraph workflow: planner → retrieval/recall/online → rank → QA.

Graph structure (updated for new Planner routing):
  planner ──→ respond ──→ END                         (NO_RETRIEVAL)
          ├─→ ask_clarify ──→ END                      (NEED_CLARIFY)
          ├─→ handle_feedback ──→ END                  (HANDLE_FEEDBACK)
          ├─→ retrieval ──→ evaluate ──┬→ fuse → rank → respond → END   (SUFFICIENT / PARTIAL)
          │                            ├→ re_retrieve → evaluate (loop, max 2 retries)
          │                            ├→ fuse → rank → respond → END   (retries exhausted, has papers)
          │                            └→ ask_clarify → END             (retries exhausted, no papers)
          └─→ online_search ──→ recall ──→ rank ──→ respond ──→ END
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from langgraph.graph import StateGraph, END

from agent.models import Paper, UserProfile

from langgraph_agents.state import WorkflowState
from langgraph_agents.retrieval_agent import RetrievalAgent
from langgraph_agents.recall_agent import RecallAgent
from langgraph_agents.online_search_agent import OnlineSearchAgent
from langgraph_agents.planner_agent import (
    PlannerAgent,
    ROUTE_NO_RETRIEVAL,
    ROUTE_RETRIEVE_LOCAL,
    ROUTE_NEED_CLARIFY,
    ROUTE_HANDLE_FEEDBACK,
)
from langgraph_agents.qa_agent import PaperQAAgent
from langgraph_agents.rank_agent import RankAgent

logger = logging.getLogger(__name__)


def build_workflow(
    retrieval_agent: Optional[RetrievalAgent] = None,
    recall_agent: Optional[RecallAgent] = None,
    online_agent: Optional[OnlineSearchAgent] = None,
    planner_agent: Optional[PlannerAgent] = None,
    rank_agent: Optional[RankAgent] = None,
    qa_agent: Optional[PaperQAAgent] = None,
    api_key: str = "",
    default_top_k: int = 10,
):
    """Build and compile the LangGraph workflow."""
    retrieval = retrieval_agent or RetrievalAgent()
    recall = recall_agent or RecallAgent(retrieval_agent=retrieval)
    online = online_agent or OnlineSearchAgent()
    planner = planner_agent or PlannerAgent(api_key=api_key)
    rank = rank_agent or RankAgent()
    qa = qa_agent or PaperQAAgent(api_key=api_key or "dummy")

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def planner_node(state: WorkflowState) -> WorkflowState:
        """Planner: query pipeline + routing + retrieval evaluation."""
        updated = planner.plan(dict(state))
        state["plan"] = updated.get("plan")
        state["planner_decision"] = updated.get("planner_decision")
        state["optimized_query"] = updated.get("optimized_query", state.get("user_query", ""))
        state["prefer_latest_papers"] = updated.get("prefer_latest_papers", False)
        state["final_query"] = updated.get("final_query", "")
        state["parent_query"] = updated.get("parent_query", "")
        state["sub_queries"] = updated.get("sub_queries", [])
        if "online_offline_fusion_ratio" in updated:
            state["online_offline_fusion_ratio"] = updated["online_offline_fusion_ratio"]
        return state

    def online_search_node(state: WorkflowState) -> WorkflowState:
        """Fetch papers from ArXiv / Semantic Scholar."""
        result = online.run(dict(state))
        state["online_search_result"] = result.get("online_search_result", [])
        return state

    def retrieval_node(state: WorkflowState) -> WorkflowState:
        """Query-based vector retrieval using optimized/final query and sub-queries."""
        query = state.get("final_query") or state.get("optimized_query") or state.get("user_query", "")
        top_k = state.get("top_k", default_top_k) * 2
        sub_queries = state.get("sub_queries", [])

        if sub_queries:
            per_sub_k = max(top_k // len(sub_queries), 5)
            all_papers = []
            seen_ids: set = set()
            for sq in sub_queries:
                sq_papers = retrieval.retrieve_by_query(sq, top_k=per_sub_k)
                for p in sq_papers:
                    pid = getattr(p, "paper_id", None) or id(p)
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        all_papers.append(p)
            main_papers = retrieval.retrieve_by_query(query, top_k=per_sub_k)
            for p in main_papers:
                pid = getattr(p, "paper_id", None) or id(p)
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
            state["retrieval_result"] = all_papers[:top_k]
        else:
            papers = retrieval.retrieve_by_query(query, top_k=top_k)
            state["retrieval_result"] = papers
        return state

    MAX_RETRIEVAL_RETRIES = 2

    def evaluate_node(state: WorkflowState) -> WorkflowState:
        """Planner evaluates retrieval quality; tracks retry count."""
        retry_count = state.get("retrieval_retry_count", 0)

        user_query = state.get("user_query", "")
        optimized_query = state.get("optimized_query", user_query)
        retrieved = state.get("retrieval_result", [])
        evaluation = planner.evaluate_retrieval(user_query, optimized_query, retrieved)
        decision = dict(state.get("planner_decision") or {})
        decision["retrieval_evaluation"] = evaluation
        state["planner_decision"] = decision

        quality = evaluation.get("quality", "SUFFICIENT")
        has_refined = bool(evaluation.get("suggested_refined_query"))
        can_retry = quality == "INSUFFICIENT" and has_refined and retry_count < MAX_RETRIEVAL_RETRIES

        if can_retry:
            state["optimized_query"] = evaluation["suggested_refined_query"]
            state["retrieval_retry_count"] = retry_count + 1
            logger.info(
                "evaluate: INSUFFICIENT (retry %d/%d), refined_query='%s'",
                retry_count + 1, MAX_RETRIEVAL_RETRIES,
                evaluation["suggested_refined_query"][:60],
            )
        elif quality == "INSUFFICIENT":
            state["retrieval_insufficient"] = True
            reason = "retries exhausted" if retry_count >= MAX_RETRIEVAL_RETRIES else "no refined query available"
            logger.info("evaluate: INSUFFICIENT — giving up (%s, retry=%d)", reason, retry_count)
        else:
            state["retrieval_insufficient"] = False

        return state

    def re_retrieve_node(state: WorkflowState) -> WorkflowState:
        """Re-retrieve with refined query after evaluation found INSUFFICIENT."""
        query = state.get("optimized_query") or state.get("user_query", "")
        top_k = state.get("top_k", default_top_k) * 2
        papers = retrieval.retrieve_by_query(query, top_k=top_k)
        state["retrieval_result"] = papers
        return state

    def recall_node(state: WorkflowState) -> WorkflowState:
        """Multi-path recall (vector + rule + ItemCF) + online/offline fusion."""
        profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
        top_k = state.get("top_k", default_top_k) * 2
        rr = recall.offline_recall(
            user=profile,
            new_papers=state.get("online_search_result"),
            top_k_override=top_k,
            prefer_recent=state.get("prefer_latest_papers", False),
        )
        offline = rr.recommended_papers
        online_papers = state.get("online_search_result", [])
        ratio = state.get("online_offline_fusion_ratio", 0.5)
        fused = recall.fuse(online_papers, offline, ratio=ratio)
        state["recall_result"] = offline
        state["recall_stats"] = rr.recall_stats
        state["fused_candidates"] = fused
        return state

    def fuse_node(state: WorkflowState) -> WorkflowState:
        """Fuse retrieval results with any online search results."""
        offline = state.get("retrieval_result", [])
        online_papers = state.get("online_search_result", [])
        ratio = state.get("online_offline_fusion_ratio", 0.5)
        fused = recall.fuse(online_papers, offline, ratio=ratio)
        state["fused_candidates"] = fused
        return state

    def rank_node(state: WorkflowState) -> WorkflowState:
        """Rerank fused candidates."""
        fused = state.get("fused_candidates", [])
        if not fused:
            state["ranked_papers"] = []
            state["final_papers"] = []
            return state
        profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
        query = state.get("optimized_query") or state.get("user_query", "") or (profile.interest_text if profile else "")
        top_k = state.get("top_k", default_top_k)
        ranked = rank.rerank(fused, query=query, user=profile, top_k=top_k)
        state["ranked_papers"] = ranked
        state["final_papers"] = ranked[:top_k]
        return state

    def respond_node(state: WorkflowState) -> WorkflowState:
        """QA: generate response based on papers in state."""
        return qa.respond(state)

    def ask_clarify_node(state: WorkflowState) -> WorkflowState:
        """QA: ask user for clarification (replaces ask_profile for general clarification)."""
        return qa.ask_profile(state)

    def handle_feedback_node(state: WorkflowState) -> WorkflowState:
        """QA: process user feedback."""
        return qa.handle_feedback(state)

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("online_search", online_search_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("re_retrieve", re_retrieve_node)
    workflow.add_node("recall", recall_node)
    workflow.add_node("fuse", fuse_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("ask_clarify", ask_clarify_node)
    workflow.add_node("handle_feedback", handle_feedback_node)

    workflow.set_entry_point("planner")

    # ------------------------------------------------------------------
    # Routing edges
    # ------------------------------------------------------------------

    def after_planner(state: WorkflowState) -> str:
        """Route based on planner's decision."""
        decision = state.get("planner_decision") or {}
        route = decision.get("route", "")

        # Also check legacy plan field for backward compat
        if not route:
            plan = state.get("plan", {})
            route = plan.get("route", ROUTE_NO_RETRIEVAL)

        if route == ROUTE_HANDLE_FEEDBACK:
            return "handle_feedback"
        if route == ROUTE_NEED_CLARIFY:
            return "ask_clarify"
        if route == ROUTE_RETRIEVE_LOCAL:
            plan = state.get("plan", {})
            if plan.get("do_online_search") or state.get("is_daily_rec"):
                return "online_search"
            return "retrieval"
        # NO_RETRIEVAL or unknown
        return "respond"

    workflow.add_conditional_edges("planner", after_planner, {
        "handle_feedback": "handle_feedback",
        "ask_clarify": "ask_clarify",
        "online_search": "online_search",
        "retrieval": "retrieval",
        "respond": "respond",
    })

    def after_online_search(state: WorkflowState) -> str:
        """After online search: go to recall (daily) or retrieval (query-based)."""
        if state.get("is_daily_rec"):
            return "recall"
        return "retrieval"

    workflow.add_conditional_edges("online_search", after_online_search, {
        "recall": "recall",
        "retrieval": "retrieval",
    })

    # retrieval → evaluate
    workflow.add_edge("retrieval", "evaluate")

    def after_evaluate(state: WorkflowState) -> str:
        """Route after evaluation:
        - SUFFICIENT / PARTIAL → fuse (normal path)
        - INSUFFICIENT + can retry (retry < max, has refined query) → re_retrieve
        - INSUFFICIENT + cannot retry + has papers → fuse (with disclaimer flag)
        - INSUFFICIENT + cannot retry + no papers → ask_clarify
        """
        decision = state.get("planner_decision") or {}
        evaluation = decision.get("retrieval_evaluation") or {}
        quality = evaluation.get("quality", "SUFFICIENT")

        if quality != "INSUFFICIENT":
            return "fuse"

        retry_count = state.get("retrieval_retry_count", 0)
        has_refined = bool(evaluation.get("suggested_refined_query"))

        if has_refined and retry_count <= MAX_RETRIEVAL_RETRIES and not state.get("retrieval_insufficient"):
            return "re_retrieve"

        if state.get("retrieval_result"):
            return "fuse"

        return "ask_clarify"

    workflow.add_conditional_edges("evaluate", after_evaluate, {
        "re_retrieve": "re_retrieve",
        "fuse": "fuse",
        "ask_clarify": "ask_clarify",
    })

    # re_retrieve loops back to evaluate for another quality check
    workflow.add_edge("re_retrieve", "evaluate")

    workflow.add_edge("fuse", "rank")
    workflow.add_edge("recall", "rank")
    workflow.add_edge("rank", "respond")

    # terminal nodes
    workflow.add_edge("respond", END)
    workflow.add_edge("ask_clarify", END)
    workflow.add_edge("handle_feedback", END)

    return workflow.compile()
