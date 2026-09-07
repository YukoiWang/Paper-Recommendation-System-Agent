"""Versioned strategy config (design §11.2 strategy memory). Loaded at process start."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

_DIR = Path(__file__).parent

POLICY_VERSIONS = {
    "playbook": "v1",
    "planner_prompt": "planner_slot_v1",
    "critic": "critic_rule_v1",
    "intent": "cascade_v1",
    "writer": "writer_templates_v1",
}

_DEFAULTS = {
    "intent_tau": 0.42,
    "intent_delta": 0.04,
    "topic_cosine": 0.75,
    "min_candidates": 3,
    "min_rerank_score": 0.0,
    "chunk_rel_keep": 3.0,
    "chunks_per_paper": 3,
    "max_pack_chunks": 12,
    "year_min": 1990,
    "year_max": 2026,
    "session_history_turns": 8,
    "cited_lru_max": 20,
    "exposure_days": 90,
    "rewrite_max": 1,
    "more_evidence_max": 1,
}


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not val:
            continue
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                out[key] = int(val) if re.fullmatch(r"-?\d+", val) else float(val)
            except ValueError:
                out[key] = val.strip('"').strip("'")
    return out


def load_thresholds() -> Dict[str, Any]:
    path = _DIR / "thresholds.yaml"
    data = dict(_DEFAULTS)
    if path.exists():
        data.update(_parse_simple_yaml(path.read_text(encoding="utf-8")))
    return data


def load_synonyms() -> Dict[str, list]:
    path = _DIR / "synonyms.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


THRESHOLDS = load_thresholds()
SYNONYMS = load_synonyms()
