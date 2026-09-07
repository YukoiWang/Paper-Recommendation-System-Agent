"""Online critic: pass / rewrite / more_evidence (design §10.2)."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from langgraph_agents.policy import THRESHOLDS

logger = logging.getLogger(__name__)


def run_critic(state: Dict[str, Any]) -> Dict[str, Any]:
    answer = state.get("response") or ""
    papers = state.get("final_papers") or state.get("ranked_papers") or []
    pack = state.get("evidence_pack") or {}
    wo = state.get("work_order") or {}
    intent = wo.get("intent") or ""
    missing = list(wo.get("missing_slots") or [])

    decision, reason = _decide(answer, papers, pack, intent, missing, wo)
    state["critic_decision"] = decision
    state["critic_reason"] = reason
    rewrite_n = int(state.get("critic_rewrite_count") or 0)
    more_n = int(state.get("critic_more_count") or 0)
    rewrite_max = int(THRESHOLDS.get("rewrite_max", 1))
    more_max = int(THRESHOLDS.get("more_evidence_max", 1))

    if decision == "rewrite" and rewrite_n < rewrite_max:
        state["critic_rewrite_count"] = rewrite_n + 1
        state["after_critic"] = "writer"
    elif decision == "more_evidence" and more_n < more_max and intent not in ("chitchat", "meta", "followup"):
        state["critic_more_count"] = more_n + 1
        state["after_critic"] = "researcher"
    else:
        state["after_critic"] = "end"
        if decision != "pass":
            state["failure_type"] = state.get("failure_type") or (
                "hallucination_risk" if decision == "rewrite" else "low_relevance"
            )
    logger.info("critic: %s (%s)", decision, reason[:80])
    return state


def _decide(
    answer: str,
    papers: List[Any],
    pack: Dict,
    intent: str,
    missing: List[str],
    wo: Dict,
) -> Tuple[str, str]:
    if missing:
        return "pass", "clarifying slots"
    if intent in ("chitchat", "meta", "feedback"):
        return "pass", "no evidence required"
    titles = _titles(papers, pack)
    if intent in ("recommend", "daily") and not papers:
        return "rewrite", "empty recommendation list"
    if intent in ("recommend", "daily") and answer and not _looks_like_list(answer) and papers:
        return "rewrite", "recommend must be an itemized list"
    if intent in ("explain", "survey") and _looks_like_pure_list(answer) and len(answer) < 800:
        return "rewrite", "explain must not be a shelf list"
    cites = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    n_papers = len(papers) or len(pack.get("papers") or [])
    if cites and n_papers:
        bad = [c for c in cites if c < 1 or c > n_papers]
        if bad:
            return "rewrite", f"citation out of range {bad}"
    invented = _invented_titles(answer, titles)
    if invented:
        return "rewrite", f"title not in pack/list: {invented[:2]}"
    if intent == "compare":
        cov = pack.get("coverage") or {}
        if cov and any(int(v.get("n_papers") or 0) < 1 for v in cov.values()):
            return "more_evidence", "compare side missing papers"
        if cites and len(cites) < 2:
            return "rewrite", "compare needs citations on both sides"
    if intent in ("explain", "compare", "survey", "factoid") and not papers and not pack.get("papers"):
        return "more_evidence", "no papers for grounded answer"
    cr = wo.get("success_criteria") or {}
    if intent in ("recommend", "daily") and papers:
        min_p = int(cr.get("min_papers") or 1)
        if len(papers) < min_p:
            return "more_evidence", "list shorter than success_criteria"
    if intent == "factoid" and papers and not cites and "不确定" not in answer and "uncertain" not in answer.lower():
        if len(answer) > 40:
            return "rewrite", "factoid needs a citation or an uncertainty statement"
    return "pass", "ok"


def _titles(papers: List[Any], pack: Dict) -> List[str]:
    out = []
    for p in papers:
        t = getattr(p, "title", None)
        if t:
            out.append(t)
    for p in pack.get("papers") or []:
        if p.get("title"):
            out.append(p["title"])
    return out


def _looks_like_list(answer: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(\d+[\.\)]|[-*])\s+", answer))


def _looks_like_pure_list(answer: str) -> bool:
    lines = [ln for ln in (answer or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    listed = sum(1 for ln in lines if re.match(r"\s*(\d+[\.\)]|[-*])\s+", ln))
    return listed >= max(3, int(0.8 * len(lines)))


def _invented_titles(answer: str, titles: List[str]) -> List[str]:
    if not titles or not answer:
        return []
    allowed = {t.lower() for t in titles}
    found = []
    for m in re.finditer(r"[\"“]([^\"”]{12,80})[\"”]", answer):
        cand = m.group(1).strip()
        if cand.lower() not in allowed and not any(cand.lower() in a or a in cand.lower() for a in allowed):
            found.append(cand)
    return found
