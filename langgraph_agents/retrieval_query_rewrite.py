"""Apply multi-label rerank diagnosis to retrieval queries (deterministic, composable)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _uniq_preserve(xs: List[str], cap: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        s = (x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def apply_multilabel_retrieval_rewrite(
    *,
    base_query: str,
    sub_queries: List[str],
    labels: List[str],
    state: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Return (main_query, sub_queries) with stacked rewrite hints based on labels."""
    q = (base_query or "").strip()
    labs = [x.strip().upper() for x in (labels or []) if isinstance(x, str)]
    labs = [x for x in labs if x in ("A", "B", "C", "D")]
    if not labs or not q:
        return q, list(sub_queries or [])

    conv = state.get("conversation_state") or {}
    profile = state.get("user_profile")
    topics = _uniq_preserve(list(conv.get("research_topics") or []), 4)
    kws = _uniq_preserve(list(conv.get("keywords") or []), 6)

    suffix_parts: List[str] = []
    if "A" in labs and (topics or kws):
        suffix_parts.append(
            "相关性补强: " + ", ".join((topics + kws)[:6])
        )
    if "B" in labs and profile is not None:
        cats = list(getattr(profile, "preferred_categories", None) or [])
        cats = _uniq_preserve([str(c) for c in cats], 5)
        if cats:
            suffix_parts.append("领域对齐: " + ", ".join(cats))
    if "C" in labs and (topics or kws):
        suffix_parts.append("消歧/限定: " + ", ".join((topics + kws)[:6]))
    if "D" in labs:
        # Retrieval-side nudge only; routing still prefers online_search for D.
        suffix_parts.append("时效/新近: recent, survey, state-of-the-art, 2024, 2025")

    if not suffix_parts:
        return q, list(sub_queries or [])

    hint = " | ".join(suffix_parts)
    new_q = f"{q}\n[retrieval_rewrite]\n{hint}"

    new_subs = []
    for sq in sub_queries or []:
        s = (sq or "").strip()
        if not s:
            continue
        new_subs.append(f"{s}\n[retrieval_rewrite]\n{hint}")
    if not new_subs:
        new_subs = list(sub_queries or [])
    return new_q, new_subs
