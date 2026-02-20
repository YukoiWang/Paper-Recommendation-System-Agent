"""Planner agent: sole decision-maker for workflow routing.

Responsibilities:
  - Classify user intent (only place in the system)
  - Check profile sufficiency
  - Detect feedback vs query
  - Decide which path the workflow should take
  - Output a structured plan with a single `route` field

Designed for future upgrade: swap rule-based logic for LLM-based planning.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.models import Paper, UserProfile, is_profile_sufficient

logger = logging.getLogger(__name__)


class Intent:
    RECOMMEND = "recommend"
    QA = "qa"
    COMPARE = "compare"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    GENERAL = "general"


def classify_intent(query: str) -> str:
    """Keyword-based intent classification. Future: replace with LLM classifier."""
    q = query.lower().strip()
    compare_kw = [
        "compare ", "difference between", " vs ", " versus ",
        "how does it differ", "how do they differ", "pros and cons", "which is better", "compared to",
    ]
    summarize_kw = [
        "summarize", "summary", "overview", "key points", "main findings", "tl;dr", "tldr", "recap",
    ]
    recommend_kw = [
        "recommend", "suggest", "find me", "latest papers", "new papers", "recent papers",
        "show me papers", "papers about", "find papers",
    ]
    explain_kw = [
        "explain", "what is ", "what are ", "how does ", "how do ", "why does ", "why do ",
        "tell me about", "describe", "elaborate", "clarify",
    ]
    for kw in compare_kw:
        if kw in q:
            return Intent.COMPARE
    for kw in summarize_kw:
        if kw in q:
            return Intent.SUMMARIZE
    for kw in recommend_kw:
        if kw in q:
            return Intent.RECOMMEND
    for kw in explain_kw:
        if kw in q:
            return Intent.EXPLAIN
    if "?" in q or any(w in q for w in ["paper", "method", "approach", "model", "technique"]):
        return Intent.QA
    return Intent.GENERAL


# Routes that the planner can output
ROUTE_ASK_PROFILE = "ask_profile"
ROUTE_HANDLE_FEEDBACK = "handle_feedback"
ROUTE_RETRIEVAL = "retrieval"
ROUTE_RECALL = "recall"
ROUTE_RESPOND = "respond"


class PlannerAgent:
    """
    Sole decision-maker. Reads state, outputs a structured plan.

    Plan fields:
      - route: which path to take (ask_profile | handle_feedback | retrieval | recall | respond)
      - do_online_search: whether to fetch from ArXiv/S2 before recall
      - prefer_latest_papers: boost recency in recall/ranking
      - intent: classified user intent
      - reasoning: text explaining the decision (for logging / future LLM upgrade)
    """

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze state and produce routing plan. Returns updated state with `plan` field.

        Future: this method can call an LLM to produce the plan instead of rules.
        """
        query = state.get("user_query", "")
        feedback = state.get("user_feedback", "")
        profile = state.get("user_profile") or UserProfile(user_id=state.get("user_id", "anonymous"))
        is_daily = state.get("is_daily_rec", False)
        history = state.get("history", [])
        cited_papers = state.get("cited_papers", {})

        plan = {
            "route": ROUTE_RESPOND,
            "do_online_search": False,
            "prefer_latest_papers": False,
            "intent": Intent.GENERAL,
            "reasoning": "",
        }

        # --- Feedback handling (highest priority) ---
        if feedback:
            plan["route"] = ROUTE_HANDLE_FEEDBACK
            plan["reasoning"] = "User provided feedback on previous recommendation."
            state["plan"] = plan
            return state

        # --- Daily recommendation flow (before query check: daily may have empty query) ---
        if is_daily:
            if not is_profile_sufficient(profile):
                plan["route"] = ROUTE_ASK_PROFILE
                plan["reasoning"] = "Daily rec but profile insufficient; need to ask user."
            else:
                plan["route"] = ROUTE_RECALL
                plan["do_online_search"] = True
                plan["prefer_latest_papers"] = True
                plan["reasoning"] = "Daily rec with sufficient profile; online search + multi-path recall."
            state["plan"] = plan
            logger.info("Planner [daily]: %s", plan)
            return state

        # --- No query and not daily: nothing to do ---
        if not query:
            plan["route"] = ROUTE_RESPOND
            plan["reasoning"] = "No query provided."
            state["plan"] = plan
            return state

        # --- Classify intent ---
        intent = classify_intent(query)
        plan["intent"] = intent
        state["qa_intent"] = intent

        # --- Detect recency preference ---
        wants_latest = any(kw in query.lower() for kw in ["latest", "new", "recent"])
        if wants_latest:
            plan["prefer_latest_papers"] = True
            state["prefer_latest_papers"] = True
            state["online_offline_fusion_ratio"] = 0.7

        # --- User query flow ---
        if intent == Intent.RECOMMEND:
            if not is_profile_sufficient(profile):
                plan["route"] = ROUTE_ASK_PROFILE
                plan["reasoning"] = "Recommend intent but profile insufficient."
            else:
                plan["route"] = ROUTE_RETRIEVAL
                plan["do_online_search"] = wants_latest
                plan["reasoning"] = "Recommend with sufficient profile; query-based retrieval."
        elif intent in (Intent.QA, Intent.EXPLAIN, Intent.COMPARE, Intent.SUMMARIZE):
            plan["route"] = ROUTE_RETRIEVAL
            plan["reasoning"] = f"Intent={intent}; need papers for context."
        elif intent == Intent.GENERAL:
            has_context = bool(cited_papers) or any(
                "[1]" in m.get("content", "") or "paper" in m.get("content", "").lower()
                for m in history if m.get("role") == "assistant"
            )
            if has_context:
                plan["route"] = ROUTE_RESPOND
                plan["reasoning"] = "General question; conversation already has paper context."
            else:
                plan["route"] = ROUTE_RETRIEVAL
                plan["reasoning"] = "General question; no context yet, need retrieval."
        else:
            plan["route"] = ROUTE_RETRIEVAL
            plan["reasoning"] = f"Fallback: intent={intent}, doing retrieval."

        state["plan"] = plan
        logger.info("Planner: route=%s intent=%s reasoning=%s", plan["route"], plan["intent"], plan["reasoning"])
        return state
