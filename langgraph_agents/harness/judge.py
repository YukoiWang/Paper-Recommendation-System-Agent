"""Optional offline LLM judge (design §14.3). Prompt lives in judges/; not used online."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

_JUDGE = Path(__file__).parent / "judges" / "intent_v1.md"


def grade_episode_llm(ep: Dict[str, Any], llm_call: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    if llm_call is None:
        return {"skipped": True}
    prompt = _JUDGE.read_text(encoding="utf-8") if _JUDGE.exists() else "Grade the episode."
    prompt += "\n\nEpisode JSON:\n" + str({
        "intent": ep.get("intent"),
        "query": ep.get("raw_query") or ep.get("query"),
        "tool_calls": ep.get("tool_calls"),
        "failure_type": ep.get("failure_type"),
    })[:2000]
    raw = llm_call(prompt)
    return {"raw": (raw or "")[:1000], "skipped": False}
