"""Playbooks: allowed tools, cascade hints (design §8.3). YAML on disk, Python fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

_DIR = Path(__file__).parent

_FALLBACK: Dict[str, Dict[str, Any]] = {
    "explain_v1": {
        "allowed": ["search_local", "search_arxiv", "gather_evidence", "rerank", "read_fulltext", "ask_user", "finish"],
        "forbidden": ["recall_multi"],
        "suggest": ["search_local", "gather_evidence"],
        "split_entities": False,
    },
    "compare_v1": {
        "allowed": ["search_local", "search_arxiv", "gather_evidence", "rerank", "read_fulltext", "ask_user", "finish"],
        "forbidden": ["recall_multi", "merged_compare_query"],
        "suggest": ["search_local_per_entity", "search_arxiv_gap", "gather_evidence"],
        "split_entities": True,
    },
    "survey_v1": {
        "allowed": ["search_local", "search_arxiv", "gather_evidence", "rerank", "finish"],
        "forbidden": ["recall_multi"],
        "suggest": ["search_local", "gather_evidence"],
        "use_sub_queries": True,
        "split_entities": False,
    },
    "factoid_v1": {
        "allowed": ["gather_evidence", "search_local", "search_arxiv", "read_fulltext", "ask_user", "finish"],
        "forbidden": ["recall_multi"],
        "suggest": ["gather_evidence"],
        "prefer_cited": True,
        "split_entities": False,
    },
    "recommend_v1": {
        "allowed": ["search_local", "search_arxiv", "recall_multi", "rerank"],
        "forbidden": [],
        "suggest": ["recall_or_search", "rerank"],
        "split_entities": False,
    },
}


def _parse_playbook_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_list: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.startswith("  - "):
            if current_list:
                data.setdefault(current_list, []).append(raw[4:].strip())
            continue
        current_list = None
        if ":" not in raw:
            continue
        key, val = raw.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val == "":
            current_list = key
            data[key] = []
        elif val.lower() in ("true", "false"):
            data[key] = val.lower() == "true"
        elif val == "[]":
            data[key] = []
        else:
            data[key] = val.strip('"').strip("'")
    return data


def _load_yaml_playbooks() -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in _FALLBACK.items()}
    if not _DIR.exists():
        return out
    for path in _DIR.glob("*.yaml"):
        parsed = _parse_playbook_yaml(path.read_text(encoding="utf-8"))
        pid = str(parsed.get("id") or path.stem)
        base = out.get(pid, {})
        base.update(parsed)
        out[pid] = base
    return out


PLAYBOOKS = _load_yaml_playbooks()


def get_playbook(playbook_id: str) -> Dict[str, Any]:
    return PLAYBOOKS.get(playbook_id) or PLAYBOOKS["explain_v1"]
