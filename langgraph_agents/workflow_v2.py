"""LangGraph v2: planner work order → recommend | researcher | writer → critic."""
from __future__ import annotations

import logging
import time
from typing import Optional

from langgraph.graph import END, StateGraph

from langgraph_agents.critic_agent import run_critic
from langgraph_agents.memory_store import MemoryStore
from langgraph_agents.planner_runtime import build_work_order, work_order_to_planner_decision
from langgraph_agents.policy import POLICY_VERSIONS
from langgraph_agents.recommend_runtime import run_recommend
from langgraph_agents.researcher_agent import run_researcher
from langgraph_agents.state import WorkflowState
from langgraph_agents.tools.registry import ToolRegistry
from langgraph_agents.writer_agent import run_writer

logger = logging.getLogger(__name__)


def build_workflow_v2(
    retrieval_agent=None,
    recall_agent=None,
    online_agent=None,
    planner_agent=None,
    rank_agent=None,
    qa_agent=None,
    api_key: str = "",
    default_top_k: int = 10,
    memory: Optional[MemoryStore] = None,
):
    tools = ToolRegistry(retrieval=retrieval_agent, recall=recall_agent, rank=rank_agent)
    store = memory or MemoryStore()
    embedder = getattr(retrieval_agent, "embedder", None)

    def _llm_call(prompt: str) -> str:
        if planner_agent is not None and getattr(planner_agent, "_llm", None):
            return planner_agent._llm.call(prompt, temperature=0.1, max_tokens=128)
        if qa_agent is not None and getattr(qa_agent, "llm", None):
            return qa_agent.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=128,
            )
        return "{}"

    llm = _llm_call if (api_key or planner_agent) else None

    def planner_node(state: WorkflowState) -> WorkflowState:
        t0 = state.get("_t0") or time.time()
        state["_t0"] = t0
        query = state.get("user_query") or ""
        cited = state.get("cited_papers") or {}
        conv = state.get("conversation_state") or {}
        order = build_work_order(
            query,
            is_daily=bool(state.get("is_daily_rec")),
            has_cited=bool(cited),
            cited=cited,
            profile=state.get("user_profile"),
            embedder=embedder,
            llm_call=llm,
            top_k=int(state.get("top_k") or default_top_k),
            last_intent=str(conv.get("last_intent") or ""),
            last_was_list=bool(state.get("last_was_list") or conv.get("last_was_list")),
            needs_replan=bool(state.get("needs_replan")),
            replan_count=int(state.get("replan_count") or 0),
        )
        if state.get("needs_replan"):
            state["replan_count"] = int(state.get("replan_count") or 0) + 1
            state["needs_replan"] = False
        state["work_order"] = order.to_state_dict()
        state["planner_decision"] = work_order_to_planner_decision(order)
        state["optimized_query"] = order.topic
        state["final_query"] = order.topic
        state["prefer_latest_papers"] = order.need_sota
        state["qa_intent"] = order.intent
        state["intent_source"] = order.intent_source
        state["policy_versions"] = dict(POLICY_VERSIONS)
        state["sub_queries"] = list((order.slots or {}).get("sub_queries") or [])
        return state

    def recommend_node(state: WorkflowState) -> WorkflowState:
        uid = state.get("user_id") or "anonymous"
        state["daily_seen_ids"] = store.recent_exposures(uid)
        return run_recommend(dict(state), tools)

    def researcher_node(state: WorkflowState) -> WorkflowState:
        return run_researcher(dict(state), tools, llm_call=llm)

    def writer_node(state: WorkflowState) -> WorkflowState:
        return run_writer(dict(state), qa_agent)

    def critic_node(state: WorkflowState) -> WorkflowState:
        st = run_critic(dict(state))
        if (st.get("after_critic") or "end") == "end":
            try:
                dt = (time.time() - float(state.get("_t0") or time.time())) * 1000
                store.persist_turn(st, latency_ms=dt)
                papers = st.get("final_papers") or []
                store.record_exposures(st.get("user_id") or "anonymous", papers)
            except Exception as e:
                logger.warning("persist_turn failed: %s", e)
        return st

    g = StateGraph(WorkflowState)
    g.add_node("planner", planner_node)
    g.add_node("recommend", recommend_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.set_entry_point("planner")

    def after_planner(state: WorkflowState) -> str:
        wo = state.get("work_order") or {}
        nxt = wo.get("next_agent") or "writer"
        if nxt in ("recommend", "researcher", "writer"):
            return nxt
        return "writer"

    g.add_conditional_edges("planner", after_planner, {
        "recommend": "recommend",
        "researcher": "researcher",
        "writer": "writer",
    })
    g.add_edge("recommend", "writer")

    def after_researcher(state: WorkflowState) -> str:
        if state.get("needs_replan") and int(state.get("replan_count") or 0) < 1:
            return "planner"
        return "writer"

    g.add_conditional_edges("researcher", after_researcher, {
        "planner": "planner",
        "writer": "writer",
    })
    g.add_edge("writer", "critic")

    def after_critic(state: WorkflowState) -> str:
        dest = state.get("after_critic") or "end"
        if dest == "writer":
            return "writer"
        if dest == "researcher":
            return "researcher"
        return "end"

    g.add_conditional_edges("critic", after_critic, {
        "writer": "writer",
        "researcher": "researcher",
        "end": END,
    })
    return g.compile()
