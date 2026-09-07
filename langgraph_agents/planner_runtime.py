"""Build a WorkOrder from query + session (design §6)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph_agents.intent.classifier import classify_intent, detect_need_sota, extract_compare_entities
from langgraph_agents.policy import POLICY_VERSIONS, THRESHOLDS
from langgraph_agents.schemas.plan import Budget, SuccessCriteria, WorkOrder
from langgraph_agents.slots import (
    apply_feedback_constraints,
    extract_expertise,
    extract_years,
    grounded_topic,
    referent_ids,
    survey_sub_queries,
)

logger = logging.getLogger(__name__)

PLAYBOOK = {
    "recommend": "recommend_v1",
    "daily": "recommend_v1",
    "explain": "explain_v1",
    "compare": "compare_v1",
    "survey": "survey_v1",
    "factoid": "factoid_v1",
}


def build_work_order(
    query: str,
    *,
    is_daily: bool = False,
    has_cited: bool = False,
    cited: Optional[Dict[str, Any]] = None,
    profile=None,
    embedder=None,
    llm_call=None,
    top_k: int = 10,
    last_intent: str = "",
    last_was_list: bool = False,
    needs_replan: bool = False,
    replan_count: int = 0,
) -> WorkOrder:
    intent, source, conf = classify_intent(
        query,
        has_cited=has_cited,
        is_daily=is_daily,
        embedder=embedder,
        llm_call=llm_call,
        tau=float(THRESHOLDS.get("intent_tau", 0.42)),
        delta=float(THRESHOLDS.get("intent_delta", 0.04)),
    )
    need_sota = detect_need_sota(query)
    year_from, year_to = extract_years(query)
    if year_from is None and need_sota:
        from datetime import datetime
        year_from = datetime.now().year - 2
    if "prefer_recent" in list(getattr(profile, "special_requirements", None) or []):
        need_sota = True

    topic = grounded_topic(query, embedder=embedder)
    expertise = extract_expertise(query, profile)

    entities = {"methods": [], "datasets": [], "authors": []}
    missing: List[str] = []
    sub_queries: List[str] = []
    referents: List[str] = []

    if intent == "compare":
        ents = extract_compare_entities(query)
        entities["methods"] = ents
        if len(ents) < 2:
            missing.append("compare_entities")

    if intent == "recommend":
        bare = not topic or topic in ("推荐论文", "推荐几篇论文", "recommend papers")
        has_profile = bool(
            profile and (getattr(profile, "interest_text", None) or getattr(profile, "preferred_categories", None))
        )
        if bare and not has_profile:
            missing.append("topic")

    if intent == "survey":
        sub_queries = survey_sub_queries(topic)

    if intent == "followup" and cited:
        referents = referent_ids(query, cited)

    if intent == "feedback" and profile is not None:
        apply_feedback_constraints(profile, query)
        need_sota = need_sota or ("prefer_recent" in (profile.special_requirements or []))

    if needs_replan:
        need_sota = True
        source = source or "rule"

    next_agent = _next_agent(
        intent, missing, has_cited,
        last_was_list=last_was_list or last_intent in ("recommend", "daily"),
    )
    criteria = _criteria(intent, entities["methods"], top_k)
    budget = Budget()
    if needs_replan:
        budget.max_search = budget.max_search + 2
        budget.max_replan = max(0, budget.max_replan - replan_count)

    order = WorkOrder(
        intent=intent,
        intent_source=source,
        confidence=conf,
        entities=entities,
        slots={
            "topic": topic,
            "need_sota": need_sota,
            "need_fulltext": False,
            "expertise": expertise,
            "year_from": year_from,
            "year_to": year_to,
            "top_k": top_k,
            "sub_queries": sub_queries,
        },
        missing_slots=missing,
        success_criteria=criteria,
        next_agent=next_agent,
        playbook_id=PLAYBOOK.get(intent, "explain_v1"),
        budget=budget,
        topic=topic,
        referent_paper_ids=referents,
        need_sota=need_sota,
        reasoning=f"intent={intent} via {source}" + (" replan" if needs_replan else ""),
    )
    logger.info("WorkOrder intent=%s source=%s next=%s missing=%s", intent, source, next_agent, missing)
    return order


def _next_agent(intent: str, missing: List[str], has_cited: bool, last_was_list: bool = False) -> str:
    if missing:
        return "writer"
    if intent in ("chitchat", "meta"):
        return "writer"
    if intent == "feedback":
        return "recommend" if last_was_list else "writer"
    if intent in ("daily", "recommend"):
        return "recommend"
    if intent == "followup" and has_cited:
        return "writer"
    return "researcher"


def _criteria(intent: str, methods: List[str], top_k: int) -> SuccessCriteria:
    if intent == "compare":
        return SuccessCriteria(
            type="compare",
            min_papers_per_entity=2,
            min_chunks_rel_ge_3_per_entity=2,
            top_k=top_k,
        )
    if intent in ("recommend", "daily"):
        return SuccessCriteria(type=intent, min_papers=min(3, top_k), top_k=top_k)
    if intent == "factoid":
        return SuccessCriteria(type="factoid", min_papers=1, min_chunks_rel_ge_3=1, top_k=top_k)
    if intent == "survey":
        return SuccessCriteria(type="survey", min_papers=4, min_chunks_rel_ge_3=4, top_k=top_k)
    return SuccessCriteria(type=intent, min_papers=3, min_chunks_rel_ge_3=3, top_k=top_k)


def work_order_to_planner_decision(order: WorkOrder) -> Dict[str, Any]:
    """Keep legacy fields so Writer/QA still compile."""
    route = "NEED_CLARIFY" if order.missing_slots else {
        "chitchat": "NO_RETRIEVAL",
        "meta": "NO_RETRIEVAL",
        "followup": "NO_RETRIEVAL",
        "feedback": "HANDLE_FEEDBACK",
        "recommend": "RETRIEVE_LOCAL",
        "daily": "RETRIEVE_LOCAL",
        "explain": "RETRIEVE_LOCAL",
        "compare": "RETRIEVE_LOCAL",
        "survey": "RETRIEVE_LOCAL",
        "factoid": "RETRIEVE_LOCAL",
    }.get(order.intent, "RETRIEVE_LOCAL")
    return {
        "route": route,
        "response_style": "recommend" if order.intent in ("recommend", "daily") else "narrative",
        "optimized_query": order.topic or "",
        "reasoning": order.reasoning,
        "intent": order.intent,
        "intent_source": order.intent_source,
        "next_agent": order.next_agent,
        "do_online_search": order.need_sota,
        "policy_versions": dict(POLICY_VERSIONS),
        "playbook_id": order.playbook_id,
    }
