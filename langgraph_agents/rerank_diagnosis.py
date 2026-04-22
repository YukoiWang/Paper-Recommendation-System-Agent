"""Hybrid rerank diagnosis.

- Deterministic layer: build evidence + rule-based diagnosis with confidence.
- LLM layer: generate natural-language suggestion/reasoning; can override label only
  when rule-confidence is low and the proposed label is consistent with evidence.

This module does NOT change ranking; it only writes diagnosis outputs to WorkflowState.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.models import Paper


def _parse_year(pub: str) -> int:
    if not pub:
        return 0
    m = re.search(r"20\d{2}", str(pub))
    return int(m.group()) if m else 0


def aggregate_rerank_score(ranked: List[Paper], top_n: int = 5) -> float:
    """Map top paper scores to [0, 1] (handles BGE logits vs LLM rank-style scores)."""
    if not ranked:
        return 0.0
    top = ranked[: max(1, min(top_n, len(ranked)))]
    scores = [float(getattr(p, "score", 0.0) or 0.0) for p in top]
    if not scores:
        return 0.0
    # LLM reranker uses discrete descending scores ~ len, len-1, ...
    if all(s == int(s) and 1 <= s <= len(ranked) + 2 for s in scores):
        return max(0.0, min(1.0, scores[0] / max(len(ranked), 1)))
    # BGE / logits: sigmoid per score then mean
    out = []
    for s in scores:
        t = max(-20.0, min(20.0, float(s)))
        out.append(1.0 / (1.0 + math.exp(-t)))
    return max(0.0, min(1.0, sum(out) / len(out)))


def _tokenize(text: str) -> set:
    return {w for w in re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()) if len(w) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _paper_card(p: Paper, abstract_chars: int = 260) -> Dict[str, Any]:
    title = (getattr(p, "title", "") or "").strip()
    abstract = (getattr(p, "abstract", "") or "").strip()
    cats = list(getattr(p, "categories", None) or [])
    pub = (getattr(p, "published", "") or "").strip()
    year = _parse_year(pub)
    if abstract and len(abstract) > abstract_chars:
        abstract = abstract[:abstract_chars].rstrip() + "..."
    return {
        "title": title[:180],
        "year": year or None,
        "categories": cats[:6],
        "abstract": abstract,
        "score": float(getattr(p, "score", 0.0) or 0.0),
    }


def _safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output, tolerating fences/extra text."""
    if not raw:
        return None
    s = raw.strip()
    # Strip markdown fences if present.
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Try direct parse.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Fallback: extract first {...} block.
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_rerank_evidence(
    user_query: str,
    optimized_query: str,
    ranked: List[Paper],
    retrieval_result: List[Paper],
    online_search_result: List[Paper],
    prefer_latest: bool,
    user_profile: Any,
) -> Dict[str, Any]:
    agg = aggregate_rerank_score(ranked)
    q = f"{user_query} {optimized_query}".strip()
    q_tokens = _tokenize(q)

    top_titles = [p.title for p in ranked[:5] if getattr(p, "title", "")]
    title_tokens = _tokenize(" ".join(top_titles))
    overlap = _jaccard(q_tokens, title_tokens)

    years = [_parse_year(getattr(p, "published", "") or "") for p in ranked[:8]]
    years = [y for y in years if y > 0]
    ref_year = 2026
    old_ratio = 0.0
    if years:
        old_ratio = sum(1 for y in years if y < ref_year - 3) / len(years)

    prof_cats: List[str] = []
    if user_profile is not None:
        prof_cats = list(getattr(user_profile, "preferred_categories", None) or [])
    cat_tokens = _tokenize(" ".join(prof_cats))
    paper_cats = _tokenize(
        " ".join(
            c
            for p in ranked[:8]
            for c in (getattr(p, "categories", None) or [])
        )
    )
    cat_j = _jaccard(cat_tokens, paper_cats) if cat_tokens else None

    return {
        "agg_score": float(agg),
        "query_title_overlap": float(overlap),
        "query_token_count": int(len(q_tokens)),
        "prefer_latest": bool(prefer_latest),
        "old_ratio": float(old_ratio),
        "user_pref_categories": prof_cats[:8],
        "paper_categories_top": list(sorted(paper_cats))[:12],
        "category_jaccard": None if cat_j is None else float(cat_j),
        "num_retrieval_candidates": int(len(retrieval_result or [])),
        "num_online_candidates": int(len(online_search_result or [])),
        "top_titles": [t[:140] for t in top_titles[:5]],
        # Concrete samples for LLM explanations (separately from each source)
        "ranked_samples": [_paper_card(p) for p in (ranked or [])[:3]],
        "retrieval_samples": [_paper_card(p) for p in (retrieval_result or [])[:2]],
        "online_samples": [_paper_card(p) for p in (online_search_result or [])[:2]],
    }


def rule_scores_from_evidence(e: Dict[str, Any]) -> Dict[str, float]:
    """Return per-label score in [0,1]. Higher = more likely."""
    agg = float(e.get("agg_score", 0.0))
    overlap = float(e.get("query_title_overlap", 0.0))
    qt = int(e.get("query_token_count", 0))
    prefer_latest = bool(e.get("prefer_latest", False))
    old_ratio = float(e.get("old_ratio", 0.0))
    cat_j = e.get("category_jaccard", None)
    n_total = int(e.get("num_retrieval_candidates", 0)) + int(e.get("num_online_candidates", 0))

    # D: outdated
    sD = 0.0
    if prefer_latest:
        sD = min(1.0, max(0.0, (old_ratio - 0.35) / 0.4)) * (1.0 if agg < 0.8 else 0.7)

    # B: domain mismatch
    sB = 0.0
    if cat_j is not None:
        cj = float(cat_j)
        sB = min(1.0, max(0.0, (0.12 - cj) / 0.12)) * (1.0 if agg < 0.75 else 0.6)

    # C: terminology ambiguity
    sC = 0.0
    if qt <= 4:
        sC = min(1.0, max(0.0, (0.16 - overlap) / 0.16)) * (1.0 if agg < 0.75 else 0.6)

    # A: relevance insufficient (default)
    sA = 0.0
    if agg < 0.55:
        sA = 0.9
    elif n_total < 3 and agg < 0.65:
        sA = 0.8
    elif agg < 0.68:
        sA = 0.65
    else:
        sA = 0.25

    return {"A": float(sA), "B": float(sB), "C": float(sC), "D": float(sD)}


def rule_diagnosis(e: Dict[str, Any]) -> Tuple[str, float]:
    """Return (diag_rule, diag_conf). Confidence is margin between top-2 scores."""
    scores = rule_scores_from_evidence(e)
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_k, best_v = items[0]
    second_v = items[1][1] if len(items) > 1 else 0.0
    margin = float(best_v - second_v)
    conf = max(0.0, min(1.0, 0.5 + margin))  # map margin to [0,1], conservative baseline
    return best_k, conf


def _is_label_consistent(label: str, e: Dict[str, Any]) -> bool:
    """Hard checks: prevent label-evidence contradictions."""
    agg = float(e.get("agg_score", 0.0))
    overlap = float(e.get("query_title_overlap", 0.0))
    qt = int(e.get("query_token_count", 0))
    prefer_latest = bool(e.get("prefer_latest", False))
    old_ratio = float(e.get("old_ratio", 0.0))
    cat_j = e.get("category_jaccard", None)

    if label == "D":
        return prefer_latest and old_ratio >= 0.45 and agg < 0.85
    if label == "B":
        return (cat_j is not None) and float(cat_j) < 0.1 and agg < 0.85
    if label == "C":
        return qt <= 5 and overlap < 0.14 and agg < 0.85
    if label == "A":
        # A is the catch-all; allow but avoid obvious contradiction
        return agg < 0.9 or overlap < 0.25
    return False


def _rule_suggestion(label: str) -> str:
    if label == "D":
        return "优先使用在线检索补充近年的论文，并提高时间过滤权重。"
    if label == "B":
        return "结合用户画像的领域偏好，收紧类别过滤或改写查询以匹配目标领域。"
    if label == "C":
        return "消解术语歧义：扩展同义词、加入任务/数据集限定词后再检索。"
    return "加强检索召回：改写查询、提高 top_k 或融合更多检索信号。"


def merge_multilabel(diag_rule: str, llm_labels: List[str], evidence: Dict[str, Any]) -> List[str]:
    """Merge rule winner + validated LLM labels (1–3), deduped."""
    out: List[str] = []
    if _is_label_consistent(diag_rule, evidence):
        out.append(diag_rule)
    for lab in llm_labels:
        if lab in ("A", "B", "C", "D") and lab not in out and _is_label_consistent(lab, evidence):
            out.append(lab)
        if len(out) >= 3:
            break
    if not out:
        out = [diag_rule]
    return out[:3]


def pick_primary(labels: List[str], evidence: Dict[str, Any]) -> str:
    """Routing primary: D prefers online_search; else highest rule score among labels."""
    scores = rule_scores_from_evidence(evidence)
    if "D" in labels and _is_label_consistent("D", evidence):
        return "D"
    best = labels[0]
    best_v = float(scores.get(best, 0.0))
    for lab in labels[1:]:
        v = float(scores.get(lab, 0.0))
        if v > best_v:
            best = lab
            best_v = v
    return best


def llm_multilabel_diagnose(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_query: str,
    optimized_query: str,
    evidence: Dict[str, Any],
    qa_feedback_for_rerank: str = "",
) -> Dict[str, Any]:
    """LLM returns multi-label diagnosis + natural language suggestion/reasoning."""
    from langgraph_agents.qa_agent import LLMClient

    client = LLMClient(api_key=api_key, base_url=base_url, model=model, temperature=0.15, max_tokens=512)
    prompt = (
        "You are a rerank diagnosis expert for an academic paper recommendation system.\n"
        "You will be given a structured evidence JSON (includes retrieval/online/ranked samples) "
        "and must output JSON only.\n"
        "Rules:\n"
        "- Do NOT fabricate paper facts; only reference titles/categories/years/abstract snippets in evidence.\n"
        "- labels: 1 to 3 items, each must be one of A,B,C,D.\n"
        "  A=relevance insufficient, B=domain mismatch, C=term ambiguity, D=stale/outdated vs recency need.\n"
        "- primary_label: one of A,B,C,D; must be included in labels. Prefer D if stale/recency mismatch is clear.\n"
        "- suggestion: 1-3 short sentences, same language as user_query.\n"
        "- reasoning: 2-6 sentences, concise, cite evidence fields/samples.\n\n"
        f"User query: {user_query}\n"
        f"Optimized query: {optimized_query}\n"
        f"QA feedback (optional): {qa_feedback_for_rerank}\n"
        f"Evidence JSON: {json.dumps(evidence, ensure_ascii=False)}\n\n"
        "Return JSON exactly:\n"
        "{\n"
        '  "labels": ["A","C"],\n'
        '  "primary_label": "C",\n'
        '  "suggestion": "text",\n'
        '  "reasoning": "text"\n'
        "}\n"
    )
    raw = client.chat([{"role": "system", "content": prompt}], temperature=0.15, max_tokens=512)
    data = _safe_json_loads(raw)
    if not data:
        return {"labels": [], "primary_label": None, "suggestion": "", "reasoning": "", "raw": raw}
    labels = data.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    labels_norm: List[str] = []
    for x in labels:
        s = str(x).strip().upper()
        if s in ("A", "B", "C", "D"):
            labels_norm.append(s)
    primary = (data.get("primary_label") or "").strip().upper()
    if primary not in ("A", "B", "C", "D"):
        primary = ""
    return {
        "labels": labels_norm[:3],
        "primary_label": primary,
        "suggestion": (data.get("suggestion") or "").strip(),
        "reasoning": (data.get("reasoning") or "").strip(),
        "raw": raw,
    }


def hybrid_diagnose_rerank(
    *,
    user_query: str,
    optimized_query: str,
    ranked: List[Paper],
    retrieval_result: List[Paper],
    online_search_result: List[Paper],
    prefer_latest: bool,
    user_profile: Any,
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    qa_feedback_for_rerank: str = "",
    tau_high: float = 0.78,
    tau_low: float = 0.55,
) -> Tuple[str, List[str], str, float, float, float, str, Dict[str, Any]]:
    """Return (primary, labels, suggestion, rerank_score, overlap, conf, reasoning, evidence)."""
    evidence = build_rerank_evidence(
        user_query=user_query,
        optimized_query=optimized_query,
        ranked=ranked,
        retrieval_result=retrieval_result,
        online_search_result=online_search_result,
        prefer_latest=prefer_latest,
        user_profile=user_profile,
    )
    diag_rule, conf = rule_diagnosis(evidence)
    suggestion = _rule_suggestion(diag_rule)
    reasoning = ""
    labels: List[str] = [diag_rule]
    primary = diag_rule

    if api_key:
        llm = llm_multilabel_diagnose(
            api_key=api_key,
            base_url=base_url,
            model=model,
            user_query=user_query,
            optimized_query=optimized_query,
            evidence=evidence,
            qa_feedback_for_rerank=qa_feedback_for_rerank,
        )
        llm_labels = list(llm.get("labels") or [])
        labels = merge_multilabel(diag_rule, llm_labels, evidence)

        llm_primary = (llm.get("primary_label") or "").strip().upper()
        if tau_high <= conf:
            primary = diag_rule
        elif tau_low <= conf < tau_high:
            if llm_primary in labels and _is_label_consistent(llm_primary, evidence):
                primary = llm_primary
            else:
                primary = pick_primary(labels, evidence)
        else:
            primary = pick_primary(labels, evidence)

        if llm.get("suggestion"):
            suggestion = str(llm["suggestion"])[:400]
        if llm.get("reasoning"):
            reasoning = str(llm["reasoning"])[:800]
    else:
        labels = merge_multilabel(diag_rule, [], evidence)
        primary = pick_primary(labels, evidence)

    agg = float(evidence.get("agg_score", 0.0))
    overlap = float(evidence.get("query_title_overlap", 0.0))
    evidence_out = dict(evidence)
    evidence_out["merged_labels"] = list(labels)
    evidence_out["primary_label"] = str(primary)
    return primary, labels, suggestion, agg, overlap, float(conf), reasoning, evidence_out
