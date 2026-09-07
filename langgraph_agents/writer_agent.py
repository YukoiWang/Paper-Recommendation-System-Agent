"""Writer: intent templates + clarify; delegates generation to PaperQAAgent."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph_agents.slots import _ordered_cited
from langgraph_agents.writer_templates import template_for

logger = logging.getLogger(__name__)

CLARIFY = {
    "topic": "你更关心哪个研究方向？我可以按主题推荐论文。",
    "compare_entities": "你想对比的两边分别是哪些方法或模型？",
}


def run_writer(state: Dict[str, Any], qa) -> Dict[str, Any]:
    wo = state.get("work_order") or {}
    missing = list(wo.get("missing_slots") or [])
    intent = wo.get("intent") or ""
    if missing:
        slot = missing[0]
        msg = CLARIFY.get(slot, "请再补充一点具体信息，例如主题或方法名。")
        history = list(state.get("history") or [])
        history.append({"role": "user", "content": state.get("user_query", "")})
        history.append({"role": "assistant", "content": msg})
        state["response"] = msg
        state["history"] = history
        return state

    if intent == "feedback" and (wo.get("next_agent") == "writer"):
        q = state.get("user_query") or ""
        state["response"] = "已记下你的偏好，下一轮推荐会尽量避开这些约束。"
        hist = list(state.get("history") or [])
        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": state["response"]})
        state["history"] = hist
        return state

    if intent == "followup":
        cited = state.get("cited_papers") or {}
        papers = _ordered_cited(cited)
        refs = list(wo.get("referent_paper_ids") or [])
        if refs:
            picked = [p for p in papers if getattr(p, "paper_id", None) in set(refs)]
            papers = picked or papers
        if papers and not state.get("final_papers"):
            state["final_papers"] = papers
            state["ranked_papers"] = papers

    if not state.get("final_papers") and state.get("response_fallback"):
        msg = state["response_fallback"]
        hist = list(state.get("history") or [])
        hist.append({"role": "user", "content": state.get("user_query", "")})
        hist.append({"role": "assistant", "content": msg})
        state["response"] = msg
        state["history"] = hist
        return state

    style = template_for(intent)
    if state.get("critic_decision") == "rewrite" and state.get("critic_reason"):
        style = style + f"\nRewrite the previous answer. Fix: {state.get('critic_reason')}"
    pack = state.get("evidence_pack") or {}
    titles = [p.get("title") for p in (pack.get("papers") or []) if p.get("title")]
    if titles:
        style += "\nAllowed paper titles only:\n- " + "\n- ".join(titles[:12])
    state["writer_style_prompt"] = style

    if qa is None:
        state["response"] = _fallback_list(state, intent)
        return state

    decision = dict(state.get("planner_decision") or {})
    if intent in ("chitchat", "meta", "followup"):
        decision["route"] = "NO_RETRIEVAL"
        if intent == "followup" and state.get("final_papers"):
            decision["route"] = "RETRIEVE_LOCAL"
            decision["response_style"] = "narrative"
    elif intent in ("recommend", "daily"):
        decision["response_style"] = "recommend"
        decision["route"] = "RETRIEVE_LOCAL"
    else:
        decision["response_style"] = "narrative"
        decision["route"] = "RETRIEVE_LOCAL"
    state["planner_decision"] = decision
    return qa.respond(state)


def _fallback_list(state: Dict[str, Any], intent: str) -> str:
    papers: List[Any] = state.get("final_papers") or []
    if not papers:
        return state.get("response") or "暂时没有可用论文。"
    lines = [f"Intent={intent}"]
    for i, p in enumerate(papers[:8], 1):
        lines.append(f"{i}. {getattr(p, 'title', '')}")
    return "\n".join(lines)
