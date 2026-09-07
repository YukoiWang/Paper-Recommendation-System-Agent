"""Harness pipeline: ingest → rule-grade → cluster → propose → replay → diff (design §14.3)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from langgraph_agents.harness.cluster import cluster_episodes
from langgraph_agents.harness.eval import summarize, write_baseline
from langgraph_agents.harness.propose import propose
from langgraph_agents.harness.replay import replay_cascade, replay_recommend_cascade, replay_routing
from langgraph_agents.memory_store import MemoryStore

_ROOT = Path(__file__).resolve().parent.parent.parent
_BASE = _ROOT / "langgraph_agents" / "harness" / "baselines"


def diff_baselines(current: dict, previous_path: Path) -> dict:
    if not previous_path.exists():
        return {"has_previous": False}
    prev = json.loads(previous_path.read_text(encoding="utf-8"))
    keys = ("salvage_rate", "n")
    delta = {}
    for k in keys:
        a, b = current.get(k), prev.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[k] = a - b
        else:
            delta[k] = {"was": b, "now": a}
    return {"has_previous": True, "delta": delta, "previous": str(previous_path)}


def run_pipeline(limit: int = 200, db: str = "", out: str = "") -> dict:
    store = MemoryStore(db or None)
    eps = store.recent_episodes(limit=limit)
    summary = summarize(eps)
    clusters = cluster_episodes(eps)
    proposals = propose(clusters)
    routing = replay_routing()
    cascade = replay_cascade()
    rec = replay_recommend_cascade()
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "clusters": {k: len(v) for k, v in clusters.items()},
        "proposals": [str(p) for p in proposals],
        "replay": {"routing": routing, "cascade": cascade, "recommend_cascade": rec},
    }
    latest = _BASE / "latest.json"
    report["diff"] = diff_baselines(summary, latest)
    dest = Path(out) if out else _BASE / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_baseline(summary, str(latest))
    report["wrote"] = str(dest)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = run_pipeline(limit=args.limit, db=args.db, out=args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
