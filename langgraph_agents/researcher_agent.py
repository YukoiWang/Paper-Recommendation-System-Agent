"""Researcher agent loop: playbook + cascade + evidence pack (design §8)."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agent.models import Paper
from langgraph_agents.playbooks import get_playbook
from langgraph_agents.policy import THRESHOLDS
from langgraph_agents.schemas.plan import EvidenceChunk, EvidencePack, SuccessCriteria
from langgraph_agents.slots import _ordered_cited
from langgraph_agents.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def run_researcher(
    state: Dict[str, Any],
    tools: ToolRegistry,
    llm_call: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    wo = state.get("work_order") or {}
    query = wo.get("topic") or state.get("user_query") or ""
    intent = wo.get("intent") or "explain"
    pb = get_playbook(wo.get("playbook_id") or "explain_v1")
    budget = wo.get("budget") or {}
    max_steps = int(budget.get("max_steps") or 8)
    max_search = int(budget.get("max_search") or 4)
    max_fulltext = int(budget.get("max_fulltext") or 2)
    criteria = SuccessCriteria(**(wo.get("success_criteria") or {"type": intent}))
    methods: List[str] = list((wo.get("entities") or {}).get("methods") or [])
    need_sota = bool(wo.get("need_sota"))
    sub_queries: List[str] = list((wo.get("slots") or {}).get("sub_queries") or [])
    allowed = set(pb.get("allowed") or [])
    forbidden = set(pb.get("forbidden") or [])

    papers: List[Paper] = []
    used_arxiv: Set[str] = set()
    searches = 0
    steps = 0
    stall = 0
    fulltext_reads = 0
    last_n = -1
    last_cov: Tuple[int, int] = (-1, -1)
    trace: List[str] = []
    last_tool = ""
    last_query = query
    ask_slot = ""

    if pb.get("prefer_cited"):
        cited = state.get("cited_papers") or {}
        papers = [p for p in _ordered_cited(cited) if hasattr(p, "paper_id")]
        if papers:
            last_tool = "cited"
            last_n = len(papers)
            trace.append(f"cited n={last_n}")

    pending_queries: List[str] = []
    if pb.get("split_entities") and methods:
        pending_queries = list(methods)
    elif pb.get("use_sub_queries") and sub_queries:
        pending_queries = list(sub_queries)
    elif not papers:
        pending_queries = [query]

    def _add(new: List[Paper]) -> None:
        nonlocal papers
        seen = {p.paper_id for p in papers}
        for p in new:
            if p.paper_id not in seen:
                papers.append(p)
                seen.add(p.paper_id)

    def _cov_tuple() -> Tuple[int, int]:
        if methods:
            return (len(papers), sum(_entity_count(papers, m) for m in methods))
        return (len(papers), 0)

    while steps < max_steps:
        steps += 1
        pack_preview_ok = _success(papers, methods, criteria, intent)
        if pack_preview_ok and last_tool in ("gather_evidence", "cited", "rerank", "finish"):
            trace.append("stop:success")
            break
        if searches >= max_search and papers and last_tool == "gather_evidence":
            trace.append("stop:max_search")
            break
        if stall >= 2:
            trace.append("stop:stall")
            state["needs_replan"] = True
            state["failure_type"] = state.get("failure_type") or "stale"
            break
        if intent == "compare" and len(methods) < 2:
            ask_slot = "compare_entities"
            trace.append("ask_user:compare_entities")
            break

        forced = _cascade_action(
            last_tool=last_tool,
            last_n=last_n,
            last_query=last_query,
            query=query,
            methods=methods,
            papers=papers,
            used_arxiv=used_arxiv,
            searches=searches,
            max_search=max_search,
            need_sota=need_sota,
            pending=pending_queries,
            pb=pb,
            criteria=criteria,
        )
        action = forced
        if action is None:
            action = _default_action(
                papers=papers,
                pending=pending_queries,
                last_tool=last_tool,
                searches=searches,
                max_search=max_search,
                intent=intent,
            )
        if action is None and llm_call and papers:
            action = _llm_pick_tool(
                llm_call, pb, papers, methods, searches, max_search, last_n, query,
            )
        if action is None:
            if papers:
                action = ("gather_evidence", query)
            else:
                trace.append("stop:no_action")
                ask_slot = "topic"
                break

        tool, arg = action
        if tool not in allowed or tool in forbidden:
            tool, arg = ("gather_evidence", query) if papers else ("search_local", query)
        if tool in ("search_local", "search_arxiv") and _merged_compare_query(arg, methods):
            arg = methods[0] if methods else arg
            trace.append("rewrite:split_compare_query")

        if tool == "search_local" and searches < max_search:
            got = tools.search_local(arg, k=12)
            _add(got)
            searches += 1
            last_n = len(got)
            last_tool = "search_local"
            last_query = arg
            if arg in pending_queries:
                pending_queries = [x for x in pending_queries if x != arg]
            trace.append(f"search_local q={arg[:40]!r} n={last_n}")
            if last_n == 0:
                state["failure_type"] = "empty_retrieval"
            continue

        if tool == "search_arxiv" and searches < max_search:
            if not need_sota and last_n and last_n >= int(THRESHOLDS.get("min_candidates", 3)) and last_tool == "search_local":
                trace.append("forbid_arxiv:local_sufficient")
                last_tool = "gather_evidence"
                continue
            got = tools.search_arxiv(arg, k=10)
            _add(got)
            used_arxiv.add(arg[:80])
            searches += 1
            last_n = len(got)
            last_tool = "search_arxiv"
            last_query = arg
            if last_n:
                state["recovery"] = "switch_arxiv"
            trace.append(f"search_arxiv q={arg[:40]!r} n={last_n}")
            continue

        if tool == "gather_evidence":
            last_tool = "gather_evidence"
            trace.append("gather_evidence")
            now = _cov_tuple()
            if now == last_cov:
                stall += 1
            else:
                stall = 0
            last_cov = now
            if _success(papers, methods, criteria, intent):
                break
            continue

        if tool == "rerank" and papers:
            papers = tools.rerank(papers, query=query, user=state.get("user_profile"), top_k=int(state.get("top_k") or 10))
            last_tool = "rerank"
            trace.append("rerank")
            continue

        if tool == "read_fulltext" and papers and fulltext_reads < max_fulltext:
            _ = tools.read_fulltext(papers[0].paper_id)
            fulltext_reads += 1
            last_tool = "read_fulltext"
            trace.append(f"read_fulltext id={papers[0].paper_id}")
            continue

        if tool == "ask_user":
            ask_slot = "topic"
            trace.append("ask_user")
            break

        if tool == "finish":
            trace.append("finish")
            break

        trace.append(f"skip:{tool}")
        break

    profile = state.get("user_profile")
    ranked = tools.rerank(papers, query=query, user=profile, top_k=int(state.get("top_k") or 10)) if papers else []
    chunks_raw = tools.gather_evidence(ranked or papers, state.get("user_query") or query)
    pack = EvidencePack(
        query=query,
        intent=intent,
        papers=[_paper_card(p, i + 1) for i, p in enumerate((ranked or papers)[:12])],
        chunks=[EvidenceChunk(**c) for c in chunks_raw],
        coverage=_coverage(papers, methods, chunks_raw),
        open_gaps=[m for m in methods if _entity_count(papers, m) < 1],
        tool_trace_summary=trace,
    )
    if not papers:
        state["failure_type"] = state.get("failure_type") or "empty_retrieval"
        state["response_fallback"] = (
            "本地和在线检索都没有找到可用论文。请换一个更具体的主题，或确认方法名拼写。"
        )

    if ask_slot:
        missing = list(wo.get("missing_slots") or [])
        if ask_slot not in missing:
            missing.append(ask_slot)
        wo["missing_slots"] = missing
        wo["next_agent"] = "writer"
        state["work_order"] = wo

    dump = pack.model_dump() if hasattr(pack, "model_dump") else pack.dict()
    state["evidence_pack"] = dump
    state["ranked_papers"] = ranked or papers
    state["final_papers"] = (ranked or papers)[: int(state.get("top_k") or 10)]
    state["retrieval_result"] = papers
    state["researcher_trace"] = trace
    state["progress_ledger"] = {
        "is_satisfied": _success(papers, methods, criteria, intent),
        "is_in_loop": stall > 0,
        "stall": stall,
        "steps": steps,
        "searches": searches,
    }
    state["plan_ledger"] = {
        "goal": query,
        "playbook": wo.get("playbook_id"),
        "entities": methods,
    }
    logger.info("researcher: papers=%s trace=%s", len(papers), trace)
    return state


def _cascade_action(
    *,
    last_tool: str,
    last_n: int,
    last_query: str,
    query: str,
    methods: List[str],
    papers: List[Paper],
    used_arxiv: Set[str],
    searches: int,
    max_search: int,
    need_sota: bool,
    pending: List[str],
    pb: Dict[str, Any],
    criteria: SuccessCriteria,
) -> Optional[Tuple[str, str]]:
    if last_tool == "search_local" and last_n == 0 and searches < max_search:
        key = (last_query or query)[:80]
        if key not in used_arxiv:
            return "search_arxiv", last_query or query
    if last_tool == "search_arxiv" and last_n == 0 and searches < max_search:
        alt = f"{query} survey" if "survey" not in (query or "").lower() else f"{query} papers"
        return "search_local", alt
    if pb.get("split_entities") and methods:
        pending_m = [m for m in methods if not _entity_count(papers, m)]
        if pending_m and searches < max_search:
            return "search_local", pending_m[0]
        short = [m for m in methods if 0 < _entity_count(papers, m) < criteria.min_papers_per_entity]
        if short and searches < max_search:
            m = short[0]
            key = m[:80]
            if key not in used_arxiv:
                return "search_arxiv", m
    if pending and searches < max_search and last_tool in ("",):
        return "search_local", pending[0]
    if need_sota and papers and "arxiv:sota" not in used_arxiv and searches < max_search and last_tool == "gather_evidence":
        return "search_arxiv", query
    return None


def _default_action(
    *,
    papers: List[Paper],
    pending: List[str],
    last_tool: str,
    searches: int,
    max_search: int,
    intent: str,
) -> Optional[Tuple[str, str]]:
    if pending and searches < max_search:
        return "search_local", pending[0]
    if not papers and searches < max_search:
        return None
    if papers and last_tool != "gather_evidence":
        return "gather_evidence", ""
    return None


def _llm_pick_tool(
    llm_call, pb, papers, methods, searches, max_search, last_n, query,
) -> Optional[Tuple[str, str]]:
    prompt = (
        "Pick ONE tool as JSON {\"tool\": \"...\", \"args\": \"query\", \"reason\": \"...\"}.\n"
        f"allowed={pb.get('allowed')} forbidden={pb.get('forbidden')}\n"
        f"n_papers={len(papers)} methods={methods} searches={searches}/{max_search} last_n={last_n}\n"
        f"question={query[:200]}\nDo not include chunk text. tool must be in allowed."
    )
    try:
        raw = (llm_call(prompt) or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        tool = str(data.get("tool") or "")
        args = str(data.get("args") or query)
        if tool in set(pb.get("allowed") or []) and tool not in set(pb.get("forbidden") or []):
            return tool, args
    except Exception as e:
        logger.warning("researcher tool LLM pick failed: %s", e)
    return None


def _merged_compare_query(q: str, methods: List[str]) -> bool:
    if len(methods) < 2:
        return False
    hits = [m for m in methods if m and m.lower() in (q or "").lower()]
    return len(hits) >= 2


def _paper_card(p: Paper, cite_n: int) -> Dict[str, Any]:
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "year": p.published,
        "authors": list(p.authors or [])[:8],
        "cite_n": cite_n,
        "abstract": (p.abstract or "")[:400],
        "rerank_score": float(getattr(p, "score", 0) or 0),
        "source": "local",
    }


def _entity_count(papers: List[Paper], entity: str) -> int:
    e = (entity or "").lower()
    n = 0
    for p in papers:
        blob = f"{p.title} {p.abstract}".lower()
        if e and e in blob:
            n += 1
    return n


def _coverage(papers: List[Paper], methods: List[str], chunks: List[dict]) -> Dict[str, Dict[str, int]]:
    out = {}
    for m in methods:
        n = _entity_count(papers, m)
        n_ch = sum(
            1 for c in chunks
            if float(c.get("relevance") or 0) >= 3
            and m.lower() in (c.get("text") or "").lower()
        )
        out[m] = {"n_papers": n, "n_chunks_rel_ge_3": n_ch}
    return out


def _success(papers: List[Paper], methods: List[str], cr: SuccessCriteria, intent: str) -> bool:
    if intent == "compare" and methods:
        return all(_entity_count(papers, m) >= cr.min_papers_per_entity for m in methods)
    if intent == "factoid":
        return len(papers) >= cr.min_papers
    return len(papers) >= cr.min_papers
