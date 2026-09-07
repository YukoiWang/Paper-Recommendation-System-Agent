"""gather_evidence: chunk papers and keep high-relevance snippets (design §9)."""
from __future__ import annotations

import re
from typing import List, Sequence

from agent.models import Paper
from langgraph_agents.chunking import split_fulltext
from langgraph_agents.policy import THRESHOLDS
from langgraph_agents.tools.read_fulltext import read_fulltext


def gather_evidence(
    papers: Sequence[Paper],
    question: str,
    max_chunks: int | None = None,
    fulltext_fn=None,
) -> List[dict]:
    keep = float(THRESHOLDS.get("chunk_rel_keep", 3))
    per_paper = int(THRESHOLDS.get("chunks_per_paper", 3))
    cap = int(max_chunks or THRESHOLDS.get("max_pack_chunks", 12))
    q = (question or "").lower()
    reader = fulltext_fn or read_fulltext
    chunks: List[dict] = []
    for i, p in enumerate(papers):
        texts = _paper_windows(p, reader)
        scored = []
        for j, item in enumerate(texts):
            rel = _token_overlap_score(q, item["text"].lower())
            scored.append({
                "chunk_id": f"{p.paper_id}#{item.get('section', 'abs')}{j}",
                "paper_id": p.paper_id,
                "cite_n": i + 1,
                "text": item["text"][:800],
                "section": item.get("section") or "abstract",
                "relevance": rel,
                "relevance_reason": "overlap with question",
            })
        scored.sort(key=lambda c: c["relevance"], reverse=True)
        kept = [c for c in scored if c["relevance"] >= keep][:per_paper]
        if not kept and scored:
            kept = scored[:1]
        chunks.extend(kept)
    chunks.sort(key=lambda c: c["relevance"], reverse=True)
    return chunks[:cap]


def _paper_windows(p: Paper, reader) -> List[dict]:
    full = getattr(p, "full_text", None) or ""
    if not full:
        try:
            full = reader(p.paper_id) or ""
        except Exception:
            full = ""
    if full and len(full) > 400:
        return split_fulltext(full, title=p.title or "")
    text = (p.abstract or p.title or "").strip()
    if not text:
        return []
    return [{"section": "abstract", "text": text[:800]}]


def _token_overlap_score(query: str, text: str) -> float:
    qt = set(re.findall(r"[a-zA-Z][\w-]*|[\u4e00-\u9fff]+", query))
    tt = set(re.findall(r"[a-zA-Z][\w-]*|[\u4e00-\u9fff]+", text))
    if not qt or not tt:
        return 0.3
    return min(5.0, 1.0 + 4.0 * len(qt & tt) / max(len(qt), 1))
