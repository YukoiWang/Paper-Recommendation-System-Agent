"""Cluster episodes by failure_type + intent (design §14.3)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from langgraph_agents.harness.eval import rule_grade


def cluster_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        flags = rule_grade(ep)
        ft = ep.get("failure_type") or (flags[0] if flags else "none")
        intent = ep.get("intent") or "?"
        buckets[f"{intent}|{ft}"].append(ep)
        for fl in flags:
            buckets[f"{intent}|{fl}"].append(ep)
    return dict(buckets)
