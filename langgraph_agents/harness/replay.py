"""Stub replay of cascade + routing gold (design §14.4)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.models import Paper
from langgraph_agents.intent.classifier import classify_intent
from langgraph_agents.planner_runtime import build_work_order
from langgraph_agents.recommend_runtime import run_recommend
from langgraph_agents.researcher_agent import run_researcher

_GOLD = Path(__file__).resolve().parent.parent.parent / "data" / "golden"


class StubTools:
    def __init__(self, local_n: int = 0, arxiv_n: int = 5):
        self.local_n = local_n
        self.arxiv_n = arxiv_n
        self.calls: List[str] = []

    def search_local(self, query: str, k: int = 10, hyde_document: str = "") -> List[Paper]:
        self.calls.append(f"search_local:{query}")
        return [_paper(f"L{i}-{query[:12]}", f"{query} local paper {i}") for i in range(self.local_n)]

    def search_arxiv(self, query: str, k: int = 10, categories=None, days: int = 365) -> List[Paper]:
        self.calls.append(f"search_arxiv:{query}")
        return [_paper(f"A{i}-{query[:12]}", f"{query} arxiv paper {i}") for i in range(self.arxiv_n)]

    def recall_multi(self, profile, query=None, online_papers=None, top_k=20, prefer_recent=False, use_itemcf=True):
        return self.search_local(query or "ml", k=top_k)

    def rerank(self, papers, query: str = "", user=None, top_k: int = 10):
        return list(papers)[:top_k]

    def gather_evidence(self, papers, question: str, max_chunks: int = 12):
        return [{
            "chunk_id": f"{p.paper_id}#abs",
            "paper_id": p.paper_id,
            "cite_n": i + 1,
            "text": p.abstract or p.title,
            "section": "abstract",
            "relevance": 4.0,
            "relevance_reason": "stub",
        } for i, p in enumerate(list(papers)[:max_chunks])]

    def read_fulltext(self, paper_id: str) -> str:
        return ""


def _paper(pid: str, title: str) -> Paper:
    return Paper(paper_id=pid, title=title, abstract=title + " abstract")


def replay_cascade(path: str = "") -> Dict[str, Any]:
    gold = json.loads(Path(path or _GOLD / "cascade.json").read_text(encoding="utf-8"))
    results = []
    for case in gold.get("cases") or []:
        tools = StubTools(local_n=int(case.get("local_n") or 0), arxiv_n=5)
        state = {
            "user_query": "explain transformers",
            "work_order": {
                "intent": "explain",
                "topic": "transformers",
                "playbook_id": "explain_v1",
                "need_sota": bool(case.get("need_sota")),
                "budget": {"max_steps": 6, "max_search": 4},
                "success_criteria": {"type": "explain", "min_papers": 1},
                "entities": {"methods": []},
            },
            "top_k": 5,
        }
        if case.get("already_arxiv"):
            state["work_order"]["need_sota"] = True
        out = run_researcher(state, tools)
        rec = out.get("recovery")
        expect = case.get("expect_recovery")
        ok = (rec or None) == (expect or None) if expect is not None else True
        if case.get("id") == "forbid_switch":
            ok = not any(c.startswith("search_arxiv") for c in tools.calls)
        if case.get("id") == "must_switch":
            ok = rec == "switch_arxiv" or any(c.startswith("search_arxiv") for c in tools.calls)
        results.append({"id": case.get("id"), "ok": bool(ok), "recovery": rec, "calls": tools.calls})
    return {"n": len(results), "passed": sum(1 for r in results if r["ok"]), "results": results}


def replay_routing(path: str = "") -> Dict[str, Any]:
    gold = json.loads(Path(path or _GOLD / "routing.json").read_text(encoding="utf-8"))
    results = []
    for case in gold.get("cases") or []:
        wo = build_work_order(
            case.get("query") or "",
            has_cited=bool(case.get("has_cited")),
            is_daily=bool(case.get("is_daily")),
            last_was_list=bool(case.get("last_was_list")),
            embedder=None,
            llm_call=None,
        )
        exp_i = case.get("expect_intent")
        exp_n = case.get("expect_next")
        forbid = set(case.get("forbid_intent") or [])
        ok = True
        if exp_i:
            ok = ok and wo.intent == exp_i
        if exp_n:
            ok = ok and wo.next_agent == exp_n
        if forbid:
            ok = ok and wo.intent not in forbid
            if "recommend" in forbid and wo.intent_source == "rule" and wo.intent == "recommend":
                ok = False
        results.append({
            "id": case.get("id"),
            "ok": ok,
            "intent": wo.intent,
            "next": wo.next_agent,
            "source": wo.intent_source,
        })
    return {"n": len(results), "passed": sum(1 for r in results if r["ok"]), "results": results}


def replay_recommend_cascade() -> Dict[str, Any]:
    tools = StubTools(local_n=0, arxiv_n=4)
    state = {
        "user_query": "recommend papers on LoRA",
        "work_order": {"intent": "recommend", "topic": "LoRA", "need_sota": False},
        "top_k": 5,
    }
    out = run_recommend(state, tools)
    ok = out.get("recovery") == "switch_arxiv" and bool(out.get("final_papers"))
    return {"ok": ok, "n_papers": len(out.get("final_papers") or [])}


def classify_only(query: str, has_cited: bool = False) -> str:
    intent, _, _ = classify_intent(query, has_cited=has_cited, embedder=None, llm_call=None)
    return intent
