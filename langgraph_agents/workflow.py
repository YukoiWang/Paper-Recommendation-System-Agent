"""LangGraph workflow: planner → parallel retrieval+search / recall → rank → QA.

Graph highlights:
  - RETRIEVE_LOCAL (non-daily): parallel_fetch (retrieval + online) → evaluate → fuse → rank
  - rank → respond | retrieval | online_search (rerank diagnosis + round/delta caps)
  - QA may route back to rank once when preference score is low
  - Retrieval evaluate → fuse only (no re_retrieve loop; reranker loop owns retries)
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
from langgraph_agents.rerank_diagnosis import hybrid_diagnose_rerank
from langgraph_agents.retrieval_query_rewrite import apply_multilabel_retrieval_rewrite

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
        # New-turn reset for rerank / QA preference loops (blackboard fields)
        state["round"] = 0
        state["last_rerank_score"] = 0.0
        state["rerank_score"] = 0.0
        state["rerank_diagnosis"] = ""
        state["rerank_labels"] = []
        state["rerank_primary"] = ""
        state["rerank_confidence"] = 0.0
        state["rerank_reasoning"] = ""
        state["rerank_evidence"] = {}
        state["rerank_suggestion"] = ""
        state["qa_preference_score"] = 0.0
        state["qa_rerank_count"] = 0
        state["qa_needs_rerank"] = False
        state["qa_feedback_for_rerank"] = ""
        state["after_rank_dest"] = "respond"
        state["retrieval_retry_count"] = 0
        state.pop("_online_then_fuse_only", None)
        state.pop("_had_prior_rerank_score", None)
        state.pop("_rank_from_qa", None)

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

    def parallel_fetch_node(state: WorkflowState) -> WorkflowState:
        """Explicit query path: local retrieval + online search (sequential, both results)."""
        retrieval_node(state)
        online_search_node(state)
        return state

    def online_search_node(state: WorkflowState) -> WorkflowState:
        """Fetch papers from ArXiv / Semantic Scholar."""
        result = online.run(dict(state))
        state["online_search_result"] = result.get("online_search_result", [])
        return state

    def retrieval_node(state: WorkflowState) -> WorkflowState:
        """Query-based vector retrieval using optimized/final query and sub-queries."""
        base_query = state.get("final_query") or state.get("optimized_query") or state.get("user_query", "")
        sub_queries = list(state.get("sub_queries") or [])
        labels = list(state.get("rerank_labels") or [])
        query, sub_queries = apply_multilabel_retrieval_rewrite(
            base_query=base_query,
            sub_queries=sub_queries,
            labels=labels,
            state=dict(state),
        )
        top_k = state.get("top_k", default_top_k) * 2

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

    def evaluate_node(state: WorkflowState) -> WorkflowState:
        """Planner evaluates retrieval quality (no re-retrieve loop; reranker owns retries)."""
        user_query = state.get("user_query", "")
        optimized_query = state.get("optimized_query", user_query)
        retrieved = state.get("retrieval_result", [])
        evaluation = planner.evaluate_retrieval(user_query, optimized_query, retrieved)
        decision = dict(state.get("planner_decision") or {})
        decision["retrieval_evaluation"] = evaluation
        state["planner_decision"] = decision

        quality = evaluation.get("quality", "SUFFICIENT")
        if quality == "INSUFFICIENT" and evaluation.get("suggested_refined_query"):
            state["optimized_query"] = evaluation["suggested_refined_query"]
            logger.info(
                "evaluate: INSUFFICIENT — applying suggested_refined_query='%s' (single pass)",
                evaluation["suggested_refined_query"][:60],
            )
        if quality == "INSUFFICIENT":
            state["retrieval_insufficient"] = True
        else:
            state["retrieval_insufficient"] = False

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
        """Rerank fused candidates + expert diagnosis + next-step routing hint."""
        from_qa = bool(state.pop("_rank_from_qa", False))
        if state.pop("qa_needs_rerank", False):
            state["qa_rerank_count"] = int(state.get("qa_rerank_count", 0)) + 1

        fused = state.get("fused_candidates", [])
        if not fused:
            state["ranked_papers"] = []
            state["final_papers"] = []
            state["rerank_score"] = 0.0
            state["rerank_diagnosis"] = "A"
            state["rerank_labels"] = ["A"]
            state["rerank_primary"] = "A"
            state["rerank_suggestion"] = "候选为空，无法排序；请尝试放宽检索条件。"
            state["after_rank_dest"] = "respond"
            return state

        profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
        base_query = (
            state.get("optimized_query")
            or state.get("user_query", "")
            or (profile.interest_text if profile else "")
        )
        qa_fb = (state.get("qa_feedback_for_rerank") or "").strip()
        if qa_fb and int(state.get("qa_rerank_count", 0)) > 0:
            query_for_rank = f"{base_query}\n[QA preference feedback for reranking]\n{qa_fb}"
        else:
            query_for_rank = base_query

        top_k = state.get("top_k", default_top_k)
        ranked = rank.rerank(fused, query=query_for_rank, user=profile, top_k=top_k)
        state["ranked_papers"] = ranked
        state["final_papers"] = ranked[:top_k]

        # Daily recommendation path: no retrieval-based rerank loop.
        if state.get("is_daily_rec"):
            user_query = state.get("user_query", "")
            optimized_query = state.get("optimized_query", user_query)
            retrieval_result = list(state.get("retrieval_result") or [])
            online_search_result = list(state.get("online_search_result") or [])
            api_key = str(state.get("api_key") or "")
            base_url = str(state.get("base_url") or "https://api.deepseek.com")
            model = str(state.get("model") or "deepseek-chat")
            primary, labels, sugg, agg, overlap, conf, reasoning, evidence = hybrid_diagnose_rerank(
                user_query=user_query,
                optimized_query=optimized_query,
                ranked=ranked,
                retrieval_result=retrieval_result,
                online_search_result=online_search_result,
                prefer_latest=bool(state.get("prefer_latest_papers", False)),
                user_profile=profile,
                api_key=api_key,
                base_url=base_url,
                model=model,
                qa_feedback_for_rerank=str(state.get("qa_feedback_for_rerank") or ""),
            )
            state["rerank_diagnosis"] = primary
            state["rerank_labels"] = list(labels or [])
            state["rerank_primary"] = str(primary)
            state["rerank_suggestion"] = sugg
            state["rerank_confidence"] = float(conf)
            state["rerank_reasoning"] = str(reasoning or "")
            state["rerank_evidence"] = dict(evidence or {})
            state["last_rerank_score"] = float(state.get("rerank_score", 0.0))
            state["rerank_score"] = float(agg)
            state["_had_prior_rerank_score"] = True
            state["after_rank_dest"] = "respond"
            return state

        user_query = state.get("user_query", "")
        optimized_query = state.get("optimized_query", user_query)
        retrieval_result = list(state.get("retrieval_result") or [])
        online_search_result = list(state.get("online_search_result") or [])
        prefer_latest = bool(state.get("prefer_latest_papers", False))

        old_rerank = state.get("rerank_score")
        had_prior = bool(state.get("_had_prior_rerank_score"))
        api_key = str(state.get("api_key") or "")
        base_url = str(state.get("base_url") or "https://api.deepseek.com")
        model = str(state.get("model") or "deepseek-chat")
        primary, labels, sugg, agg, overlap, conf, reasoning, evidence = hybrid_diagnose_rerank(
            user_query=user_query,
            optimized_query=optimized_query,
            ranked=ranked,
            retrieval_result=retrieval_result,
            online_search_result=online_search_result,
            prefer_latest=prefer_latest,
            user_profile=profile,
            api_key=api_key,
            base_url=base_url,
            model=model,
            qa_feedback_for_rerank=str(state.get("qa_feedback_for_rerank") or ""),
        )
        state["rerank_diagnosis"] = primary
        state["rerank_labels"] = list(labels or [])
        state["rerank_primary"] = str(primary)
        state["rerank_suggestion"] = sugg
        state["rerank_confidence"] = float(conf)
        state["rerank_reasoning"] = str(reasoning or "")
        state["rerank_evidence"] = dict(evidence or {})
        state["last_rerank_score"] = float(old_rerank) if had_prior else float(state.get("last_rerank_score", 0.0))
        state["rerank_score"] = float(agg)
        state["_had_prior_rerank_score"] = True

        round_n = int(state.get("round", 0))
        dest = "respond"
        if had_prior and old_rerank is not None and abs(float(agg) - float(old_rerank)) < 0.05:
            dest = "respond"
            logger.info("after_rank: no material gain vs prior score → QA")
        elif round_n >= 2:
            dest = "respond"
            logger.info("after_rank: round cap → QA")
        elif float(agg) >= 0.77 or (float(agg) >= 0.68 and float(overlap) >= 0.14):
            dest = "respond"
            logger.info("after_rank: strong relevance → QA (diag=%s)", primary)
        elif not from_qa and primary in ("A", "B", "C"):
            state["round"] = round_n + 1
            dest = "retrieval"
            logger.info("after_rank: diagnosis %s → retrieval (round=%s)", primary, state["round"])
        elif not from_qa and primary == "D":
            state["round"] = round_n + 1
            dest = "online_search"
            state["_online_then_fuse_only"] = True
            logger.info("after_rank: diagnosis D → online_search (round=%s)", state["round"])
        else:
            dest = "respond"

        # QA feedback loop: only one re-rank; do not chain into retrieval/online diagnosis loop.
        if from_qa:
            dest = "respond"

        state["after_rank_dest"] = dest
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

    def no_retrieval_gate_node(state: WorkflowState) -> WorkflowState:
        """For NO_RETRIEVAL answers, decide whether to fallback to retrieval.

        Signals (written by PaperQAAgent.respond):
          - no_retrieval_confidence: float in [0,1] from logprobs, or None
          - no_retrieval_logprobs_available: bool
          - no_retrieval_verbal_flag: bool, True if model emitted [[NEED_RETRIEVAL]]
        """
        decision = state.get("planner_decision") or {}
        plan = state.get("plan") or {}
        route = decision.get("route", ROUTE_NO_RETRIEVAL)

        # Safety guard: only act when planner originally chose NO_RETRIEVAL and we
        # truly have no retrieved papers in this pass.
        if route != ROUTE_NO_RETRIEVAL:
            return state
        if state.get("final_papers") or state.get("retrieval_result") or state.get("fused_candidates"):
            return state

        # Configuration: threshold can be overridden via state.
        threshold = float(state.get("no_retrieval_confidence_threshold", 0.6))
        conf = state.get("no_retrieval_confidence")
        has_logprobs = bool(state.get("no_retrieval_logprobs_available", False))
        verbal_flag = bool(state.get("no_retrieval_verbal_flag", False))
        query = state.get("user_query", "")
        answer = state.get("response", "")

        # Method 3: explicit verbalized uncertainty wins immediately.
        if verbal_flag or ("[[NEED_RETRIEVAL]]" in (answer or "")):
            decision["route"] = ROUTE_RETRIEVE_LOCAL
            plan["route"] = ROUTE_RETRIEVE_LOCAL
            state["planner_decision"] = decision
            state["plan"] = plan
            state["forced_retrieval_reason"] = "verbal_flag"
            return state

        # Method 1: logprobs-based confidence when available.
        if has_logprobs and conf is not None:
            state["forced_retrieval_reason"] = "logprob_ok"
            if conf < threshold:
                decision["route"] = ROUTE_RETRIEVE_LOCAL
                plan["route"] = ROUTE_RETRIEVE_LOCAL
                state["forced_retrieval_reason"] = f"logprob_below_threshold:{conf:.3f}"
            state["planner_decision"] = decision
            state["plan"] = plan
            return state

        # Method 2: self-reflection evaluation as fallback when logprobs missing.
        try:
            label = qa.evaluate_no_retrieval_answer(query=query, answer=answer)
        except Exception as e:
            logger.warning("no_retrieval_gate: self-reflection failed: %s", e)
            label = "HIGH_CONFIDENCE"

        if isinstance(label, str) and label.upper().strip() == "LOW_CONFIDENCE":
            decision["route"] = ROUTE_RETRIEVE_LOCAL
            plan["route"] = ROUTE_RETRIEVE_LOCAL
            state["forced_retrieval_reason"] = "self_reflection_low"
        else:
            state["forced_retrieval_reason"] = "self_reflection_high"

        state["planner_decision"] = decision
        state["plan"] = plan
        return state

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("parallel_fetch", parallel_fetch_node)
    workflow.add_node("online_search", online_search_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("recall", recall_node)
    workflow.add_node("fuse", fuse_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("no_retrieval_gate", no_retrieval_gate_node)
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
            if state.get("is_daily_rec"):
                return "online_search"
            return "parallel_fetch"
        # NO_RETRIEVAL or unknown
        return "respond"

    workflow.add_conditional_edges("planner", after_planner, {
        "handle_feedback": "handle_feedback",
        "ask_clarify": "ask_clarify",
        "online_search": "online_search",
        "parallel_fetch": "parallel_fetch",
        "respond": "respond",
    })

    workflow.add_edge("parallel_fetch", "evaluate")

    def after_online_search(state: WorkflowState) -> str:
        """After online search: go to recall (daily) or retrieval (query-based)."""
        if state.pop("_online_then_fuse_only", None):
            return "fuse"
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
        """Route after evaluation: always fuse when any papers exist, else clarify."""
        if state.get("retrieval_result"):
            return "fuse"
        return "ask_clarify"

    workflow.add_conditional_edges("evaluate", after_evaluate, {
        "fuse": "fuse",
        "ask_clarify": "ask_clarify",
    })

    workflow.add_edge("fuse", "rank")
    workflow.add_edge("recall", "rank")

    def route_after_rank(state: WorkflowState) -> str:
        return str(state.get("after_rank_dest") or "respond")

    workflow.add_conditional_edges("rank", route_after_rank, {
        "respond": "respond",
        "retrieval": "retrieval",
        "online_search": "online_search",
    })

    def after_respond(state: WorkflowState) -> str:
        """After QA responds, decide whether to run NO_RETRIEVAL gate.

        - If this was a retrieval/recall path (papers present) → END.
        - If planner route was NO_RETRIEVAL and there are no papers → gate.
        """
        if state.get("qa_needs_rerank"):
            return "rank"

        decision = state.get("planner_decision") or {}
        route = decision.get("route", ROUTE_NO_RETRIEVAL)

        if state.get("final_papers") or state.get("ranked_papers") or state.get("retrieval_result") or state.get("fused_candidates"):
            return "end"

        if route == ROUTE_NO_RETRIEVAL:
            return "no_retrieval_gate"

        return "end"

    workflow.add_conditional_edges("respond", after_respond, {
        "rank": "rank",
        "no_retrieval_gate": "no_retrieval_gate",
        "end": END,
    })

    def after_no_retrieval_gate(state: WorkflowState) -> str:
        """After NO_RETRIEVAL gate, either fallback to retrieval or finish."""
        decision = state.get("planner_decision") or {}
        route = decision.get("route", ROUTE_NO_RETRIEVAL)

        if route == ROUTE_RETRIEVE_LOCAL:
            if state.get("is_daily_rec"):
                return "online_search"
            return "parallel_fetch"

        return "end"

    workflow.add_conditional_edges("no_retrieval_gate", after_no_retrieval_gate, {
        "online_search": "online_search",
        "parallel_fetch": "parallel_fetch",
        "end": END,
    })

    # terminal nodes
    workflow.add_edge("ask_clarify", END)
    workflow.add_edge("handle_feedback", END)

    return workflow.compile()
