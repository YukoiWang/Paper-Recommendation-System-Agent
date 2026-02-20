"""LangGraph workflow: planner → retrieval/recall/online → rank → QA.

Graph structure:
  planner ──→ ask_profile ──→ END
          ├─→ handle_feedback ──→ END
          ├─→ online_search ──→ recall ──→ rank ──→ respond ──→ END
          ├─→ retrieval ──→ fuse ──→ rank ──→ respond ──→ END
          └─→ respond ──→ END  (general with existing context)
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
    ROUTE_ASK_PROFILE,
    ROUTE_HANDLE_FEEDBACK,
    ROUTE_RETRIEVAL,
    ROUTE_RECALL,
    ROUTE_RESPOND,
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
    planner = planner_agent or PlannerAgent()
    rank = rank_agent or RankAgent()
    qa = qa_agent or PaperQAAgent(api_key=api_key or "dummy")

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def planner_node(state: WorkflowState) -> WorkflowState:
        """Planner: sole decision-maker. Sets plan with route."""
        updated = planner.plan(dict(state))
        state["plan"] = updated["plan"]
        state["qa_intent"] = updated.get("qa_intent", state.get("qa_intent", ""))
        state["prefer_latest_papers"] = updated.get("prefer_latest_papers", False)
        if "online_offline_fusion_ratio" in updated:
            state["online_offline_fusion_ratio"] = updated["online_offline_fusion_ratio"]
        return state

    def online_search_node(state: WorkflowState) -> WorkflowState:
        """Fetch papers from ArXiv / Semantic Scholar."""
        result = online.run(dict(state))
        state["online_search_result"] = result.get("online_search_result", [])
        return state

    def retrieval_node(state: WorkflowState) -> WorkflowState:
        """Query-based vector retrieval."""
        query = state.get("user_query", "")
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
        query = state.get("user_query", "") or (profile.interest_text if profile else "")
        top_k = state.get("top_k", default_top_k)
        ranked = rank.rerank(fused, query=query, user=profile, top_k=top_k)
        state["ranked_papers"] = ranked
        state["final_papers"] = ranked[:top_k]
        return state

    def respond_node(state: WorkflowState) -> WorkflowState:
        """QA: generate response based on papers in state."""
        return qa.respond(state)

    def ask_profile_node(state: WorkflowState) -> WorkflowState:
        """QA: ask user about interests."""
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
    workflow.add_node("recall", recall_node)
    workflow.add_node("fuse", fuse_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("ask_profile", ask_profile_node)
    workflow.add_node("handle_feedback", handle_feedback_node)

    workflow.set_entry_point("planner")

    # ------------------------------------------------------------------
    # Routing edges
    # ------------------------------------------------------------------

    def after_planner(state: WorkflowState) -> str:
        """Route based on planner's decision."""
        plan = state.get("plan", {})
        route = plan.get("route", ROUTE_RESPOND)
        if route == ROUTE_ASK_PROFILE:
            return "ask_profile"
        if route == ROUTE_HANDLE_FEEDBACK:
            return "handle_feedback"
        if route == ROUTE_RECALL:
            if plan.get("do_online_search"):
                return "online_search"
            return "recall"
        if route == ROUTE_RETRIEVAL:
            if plan.get("do_online_search"):
                return "online_search_then_retrieval"
            return "retrieval"
        return "respond"

    workflow.add_conditional_edges("planner", after_planner, {
        "ask_profile": "ask_profile",
        "handle_feedback": "handle_feedback",
        "online_search": "online_search",
        "online_search_then_retrieval": "online_search",
        "recall": "recall",
        "retrieval": "retrieval",
        "respond": "respond",
    })

    def after_online_search(state: WorkflowState) -> str:
        """After online search: go to recall or retrieval depending on plan."""
        plan = state.get("plan", {})
        route = plan.get("route", ROUTE_RECALL)
        if route == ROUTE_RETRIEVAL:
            return "retrieval"
        return "recall"

    workflow.add_conditional_edges("online_search", after_online_search, {
        "recall": "recall",
        "retrieval": "retrieval",
    })

    # retrieval → fuse → rank → respond
    workflow.add_edge("retrieval", "fuse")
    workflow.add_edge("fuse", "rank")

    # recall → rank → respond
    workflow.add_edge("recall", "rank")

    # rank always leads to respond
    workflow.add_edge("rank", "respond")

    # terminal nodes
    workflow.add_edge("respond", END)
    workflow.add_edge("ask_profile", END)
    workflow.add_edge("handle_feedback", END)

    return workflow.compile()
