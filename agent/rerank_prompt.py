"""Prompt templates and parsing utilities for LLM reranking."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from agent.models import Paper, UserProfile


RERANK_SYSTEM_PROMPT = """You are an expert academic paper recommender and ranker.

## Goal
Given a user's profile, the user's current query, and a list of candidate papers, produce a relevance ranking.

## What to optimize
- Relevance to the user's query intent
- Match to user's interests and preferred categories
- Match to followed authors (if any)
- Paper quality signals in title/abstract (clarity, novelty, solid method)

## Output requirements
- Return STRICT JSON only (no markdown, no extra text).
- Provide a ranking AND concise user-facing reasons.
- Do NOT reveal hidden chain-of-thought. Reasons must be short, factual, and directly grounded in the provided text.
"""


def _build_user_profile_text(user: UserProfile) -> str:
    parts: List[str] = []
    if user.interest_text:
        parts.append(f"- Interests: {user.interest_text}")
    if user.preferred_categories:
        parts.append(f"- Preferred categories: {', '.join(user.preferred_categories)}")
    if user.followed_authors:
        parts.append(f"- Followed authors: {', '.join(user.followed_authors[:10])}")
    if user.liked_items:
        parts.append(f"- Liked paper IDs (sample): {', '.join(user.liked_items[:5])}")
    return "\n".join(parts) if parts else "- (No profile / cold user)"


def _build_candidates_block(papers: List[Paper], max_abstract_chars: int = 300) -> str:
    chunks: List[str] = []
    for i, p in enumerate(papers):
        abstract = (p.abstract or "")[:max_abstract_chars]
        if (p.abstract or "") and len(p.abstract) > max_abstract_chars:
            abstract += "..."
        authors = ", ".join((p.authors or [])[:4]) or "(unknown)"
        cats = ", ".join((p.categories or [])[:5]) or "(unknown)"
        chunks.append(
            f"### Candidate {i+1}\n"
            f"- Title: {p.title}\n"
            f"- Abstract: {abstract}\n"
            f"- Authors: {authors}\n"
            f"- Categories: {cats}\n"
        )
    return "\n".join(chunks)


def build_rerank_messages(
    user: UserProfile,
    query: str,
    papers: List[Paper],
    max_abstract_chars: int = 300,
) -> List[Dict[str, str]]:
    """Build chat messages for LLM reranking."""
    profile_text = _build_user_profile_text(user)
    candidates_text = _build_candidates_block(papers, max_abstract_chars=max_abstract_chars)

    user_prompt = f"""## Task
Re-rank the candidate papers for the user.

## User profile
{profile_text}

## User query
{query}

## Candidate papers
{candidates_text}

## Output JSON schema (STRICT)
{{
  "order": [3, 1, 5, 2, 4],                // 1-based candidate indices, length = number of candidates
  "reasons": {{
    "3": ["reason 1", "reason 2"],
    "1": ["reason 1", "reason 2"],
    "...": ["..."]
  }},
  "summary": "one short sentence summarizing ranking criteria"
}}

Rules:
- "order" MUST contain each index exactly once.
- "reasons" MUST provide 1-3 concise reasons per ranked candidate.
- Reasons must be directly supported by the provided title/abstract/categories/user profile.
- Do NOT include chain-of-thought or step-by-step internal reasoning.
"""

    return [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_rerank_response(response_text: str, n_candidates: int) -> Tuple[List[int], Dict[int, List[str]], str]:
    """
    Parse LLM response.
    Returns:
      - ordered_indices: 0-based indices
      - reasons_map: {0-based idx: [reasons]}
      - summary: short string
    """
    text = (response_text or "").strip()

    # 1) Try strict JSON
    try:
        obj = json.loads(text)
        order_raw = obj.get("order", [])
        summary = str(obj.get("summary", "") or "").strip()
        reasons_raw = obj.get("reasons", {}) or {}

        ordered_0: List[int] = []
        seen = set()
        for x in order_raw:
            try:
                idx1 = int(x)
            except Exception:
                continue
            idx0 = idx1 - 1
            if 0 <= idx0 < n_candidates and idx0 not in seen:
                ordered_0.append(idx0)
                seen.add(idx0)

        # Fill missing (robustness)
        if len(ordered_0) < n_candidates:
            for i in range(n_candidates):
                if i not in seen:
                    ordered_0.append(i)
                    seen.add(i)

        reasons_map: Dict[int, List[str]] = {}
        if isinstance(reasons_raw, dict):
            for k, v in reasons_raw.items():
                try:
                    idx0 = int(k) - 1
                except Exception:
                    continue
                if not (0 <= idx0 < n_candidates):
                    continue
                if isinstance(v, list):
                    reasons_map[idx0] = [str(s).strip() for s in v if str(s).strip()][:3]
                elif isinstance(v, str) and v.strip():
                    reasons_map[idx0] = [v.strip()][:3]

        return ordered_0, reasons_map, summary
    except Exception:
        pass

    # 2) Fallback: extract numbers for order
    nums = re.findall(r"\d+", text)
    ordered_0: List[int] = []
    seen = set()
    for num in nums:
        idx0 = int(num) - 1
        if 0 <= idx0 < n_candidates and idx0 not in seen:
            ordered_0.append(idx0)
            seen.add(idx0)
    if not ordered_0:
        ordered_0 = list(range(n_candidates))
    if len(ordered_0) < n_candidates:
        for i in range(n_candidates):
            if i not in seen:
                ordered_0.append(i)
                seen.add(i)

    return ordered_0, {}, ""

