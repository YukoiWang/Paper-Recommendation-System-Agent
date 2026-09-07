"""Write executable harness proposals (design §14.3 Propose). Never auto-merges."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_PROPOSALS = Path(__file__).resolve().parent / "proposals"


def propose(clusters: Dict[str, List[Dict[str, Any]]], out_dir: str = "") -> List[Path]:
    root = Path(out_dir) if out_dir else _PROPOSALS
    root.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    day = datetime.now().strftime("%Y-%m-%d")
    n = 0
    for key, eps in clusters.items():
        intent, kind = (key.split("|", 1) + ["none"])[:2]
        if kind in ("none", "") or len(eps) < 2:
            continue
        n += 1
        pid = f"{day}-{n:03d}"
        change, target, expected = _change_for(kind)
        body = (
            f"id: {pid}\n"
            f"target: {target}\n"
            f"type: threshold\n"
            f"change:\n"
            f"  {change}\n"
            f"reason: \"{len(eps)} episodes clustered as {key}\"\n"
            f"expected_metric: {expected}\n"
            f"auto_merge: false\n"
        )
        path = root / f"{pid}.yaml"
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def _change_for(kind: str) -> tuple:
    if kind == "cascade_miss":
        return (
            'empty_local_definition: "n < 3 or max_score < 0.4"',
            "langgraph_agents/policy/thresholds.yaml",
            "salvage_rate",
        )
    if kind == "over_retrieve":
        return (
            "followup_must_skip_search_local: true",
            "langgraph_agents/playbooks/factoid_v1.yaml",
            "over_retrieve_rate",
        )
    if kind == "slot_invalid":
        return (
            "forbid_merged_compare_query: true",
            "langgraph_agents/playbooks/compare_v1.yaml",
            "compare_coverage",
        )
    if kind == "hallucination_risk":
        return (
            "critic_block_pack_external_titles: true",
            "langgraph_agents/critic_agent.py",
            "grounding_pass_rate",
        )
    return (f"review_cluster: {kind}", "langgraph_agents/policy/thresholds.yaml", "manual")
