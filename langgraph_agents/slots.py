"""Slot filling and grounding after intent is known (design §4)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph_agents.intent.classifier import topic_from_query, topic_is_grounded
from langgraph_agents.policy import SYNONYMS, THRESHOLDS

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_EXPERT = re.compile(r"(beginner|入门|小白|expert|专家|researcher)", re.I)
_FOLLOW_N = re.compile(
    r"(第([一二三四五12345])篇|the (first|second|third|1st|2nd|3rd) paper)",
    re.I,
)
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
_EN_NUM = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3}


def extract_years(query: str) -> Tuple[Optional[int], Optional[int]]:
    years = [int(x.group(0)) for x in _YEAR.finditer(query or "")]
    years = [y for y in years if THRESHOLDS["year_min"] <= y <= THRESHOLDS["year_max"]]
    if not years:
        return None, None
    return min(years), max(years)


def extract_expertise(query: str, profile=None) -> str:
    m = _EXPERT.search(query or "")
    if m:
        t = m.group(1).lower()
        if t in ("beginner", "入门", "小白"):
            return "beginner"
        if t in ("expert", "专家", "researcher"):
            return "expert"
    lvl = getattr(profile, "expertise_level", None) if profile is not None else None
    if lvl in ("beginner", "expert", "unknown"):
        return lvl
    if lvl in ("intermediate", "researcher"):
        return "expert" if lvl == "researcher" else "beginner"
    return "unknown"


def parse_followup_index(query: str) -> Optional[int]:
    m = _FOLLOW_N.search(query or "")
    if not m:
        return None
    blob = m.group(0).lower()
    for k, n in _EN_NUM.items():
        if k in blob:
            return n
    for k, n in _CN_NUM.items():
        if k in blob:
            return n
    return None


def referent_ids(query: str, cited: Dict[str, Any]) -> List[str]:
    if not cited:
        return []
    idx = parse_followup_index(query)
    ordered = _ordered_cited(cited)
    if idx and 1 <= idx <= len(ordered):
        p = ordered[idx - 1]
        return [getattr(p, "paper_id", str(idx))]
    return []


def _ordered_cited(cited: Dict[str, Any]) -> List[Any]:
    items = list(cited.items())

    def _key(kv):
        k = kv[0]
        digits = re.findall(r"\d+", str(k))
        return int(digits[0]) if digits else 0

    items.sort(key=_key)
    return [v for _, v in items]


def survey_sub_queries(topic: str, max_n: int = 3) -> List[str]:
    t = (topic or "").strip()
    if not t:
        return []
    parts = re.split(r"[,，;/]|以及|和|与|&", t)
    qs = [p.strip() for p in parts if 2 < len(p.strip()) <= 80]
    if len(qs) >= 2:
        return qs[:max_n]
    return [t]


def grounded_topic(query: str, embedder=None) -> str:
    topic = topic_from_query(query)
    syn = {}
    for k, vs in SYNONYMS.items():
        syn[k] = vs[0] if vs else k
        for v in vs:
            syn[v.lower()] = k
    if not topic_is_grounded(topic, query, synonyms=syn):
        return (query or "").strip()
    if embedder is not None and topic and query:
        try:
            import numpy as np
            qv = np.asarray(embedder.encode(query), dtype=np.float32).ravel()
            tv = np.asarray(embedder.encode(topic), dtype=np.float32).ravel()
            qn = float(np.linalg.norm(qv)) or 1.0
            tn = float(np.linalg.norm(tv)) or 1.0
            cos = float((qv / qn) @ (tv / tn))
            if cos < float(THRESHOLDS.get("topic_cosine", 0.75)):
                return (query or "").strip()
        except Exception:
            pass
    if len(topic) > max(len(query or ""), 8) * 2:
        return (query or "").strip()
    return topic


def apply_feedback_constraints(profile, query: str) -> None:
    if profile is None:
        return
    q = query or ""
    reqs = list(getattr(profile, "special_requirements", None) or [])
    if re.search(r"太老|too old|不要老", q, re.I):
        if "prefer_recent" not in reqs:
            reqs.append("prefer_recent")
    if re.search(r"不要综述|no survey|don't like survey", q, re.I):
        if "no_survey" not in reqs:
            reqs.append("no_survey")
    if re.search(r"不喜欢|换一批|not these", q, re.I):
        if q[:120] not in reqs:
            reqs.append(q[:120])
    profile.special_requirements = reqs[-12:]
