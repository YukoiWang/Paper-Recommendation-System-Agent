"""Harness: rule-grade episodes, salvage rate, optional replay (design §14)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def rule_grade(ep: Dict[str, Any]) -> List[str]:
    flags = []
    tools = ep.get("tool_calls")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            tools = []
    tools = tools or []
    joined = " ".join(str(t) for t in tools)
    ft = (ep.get("failure_type") or "").lower()
    if ft == "empty_retrieval" and "arxiv" not in joined.lower() and "force_arxiv" not in joined and "search_arxiv" not in joined:
        flags.append("cascade_miss")
    if (ep.get("intent") == "followup") and "search_local" in joined:
        flags.append("over_retrieve")
    if (ep.get("intent") == "compare") and (" vs " in joined.lower() or " versus " in joined.lower()):
        flags.append("slot_invalid")
    if (ep.get("critic") or ep.get("critic_decision")) == "pass" and (ep.get("failure_type") or "") == "hallucination_risk":
        flags.append("hallucination_risk")
    return flags


def salvage_rate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    empty = 0
    saved = 0
    for ep in episodes:
        tools = ep.get("tool_calls")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                tools = []
        blob = " ".join(str(t) for t in (tools or []))
        if (ep.get("failure_type") or "") == "empty_retrieval" or "search_local n=0" in blob:
            empty += 1
            if ep.get("recovery") == "switch_arxiv" or "arxiv" in blob.lower():
                saved += 1
    return {
        "empty_first": empty,
        "saved": saved,
        "salvage_rate": (saved / empty) if empty else None,
    }


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    flags = []
    for ep in episodes:
        flags.extend(rule_grade(ep))
    return {
        "n": len(episodes),
        "intents": dict(Counter(ep.get("intent") or "?" for ep in episodes)),
        "rule_flags": dict(Counter(flags)),
        **salvage_rate(episodes),
    }


def write_baseline(summary: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
