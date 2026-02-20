"""
Recall node (LangGraph-style) for a paper recommendation system.

What you get:
- SystemState + Candidate dataclasses (minimal but extensible)
- Pluggable RecallSource interface
- 3 recall sources (Vector / BM25 / Fresh) with clean stubs you can wire to your own stores
- Score normalization + merge/dedupe + recall_score aggregation + light filtering
- A single `recall_node(state)` function that updates:
    - state.candidates_recall
    - state.recall_trace

You can drop this into your project and replace the TODO parts that call your vector DB / BM25 / storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Protocol
import math
import time
import re
import hashlib


# -----------------------------
# Data models
# -----------------------------

@dataclass
class UserProfile:
    user_id: str
    interest_embedding: Optional[List[float]] = None  # user preference vector
    preferred_keywords: List[str] = field(default_factory=list)
    author_affinity: List[str] = field(default_factory=list)
    venue_affinity: List[str] = field(default_factory=list)
    recent_history: Dict[str, List[str]] = field(default_factory=dict)  # clicked/saved/skipped paper_ids


@dataclass
class QueryContext:
    mode: str  # "daily" | "interactive"
    user_query: Optional[str] = None
    generated_queries: List[str] = field(default_factory=list)


@dataclass
class RecallConfig:
    per_source_topk: Dict[str, int] = field(default_factory=lambda: {
        "vec": 200,
        "bm25": 200,
        "fresh": 100,
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        "w_vec": 0.45,
        "w_bm25": 0.35,
        "w_fresh": 0.20,
        "w_cf": 0.0,
        "bonus_multi": 0.05,  # bonus if recalled by >=2 sources
        "bonus_multi_cap": 0.10,
    })
    filters: Dict[str, Any] = field(default_factory=lambda: {
        "max_age_days": 3650,               # allow old, filter later in ranking if you want
        "language": None,                   # e.g. "en" if you have language metadata
        "min_title_len": 8,
        "drop_if_missing_abstract": False,
        "final_cap": 500,                   # how many to pass to ranking
    })


@dataclass
class Resources:
    # Put your handles/clients here (vector index client, BM25 index handle, etc.)
    vector_index: Any = None
    bm25_index: Any = None
    # Optional future:
    cf_sim_table: Any = None


@dataclass
class Candidate:
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    published_at: Optional[datetime] = None
    url: Optional[str] = None

    # recall metadata
    sources: Set[str] = field(default_factory=set)
    scores_by_source: Dict[str, float] = field(default_factory=dict)
    recall_score: float = 0.0
    reasons: List[str] = field(default_factory=list)

    # lightweight features for ranker
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallStats:
    requested_k: int
    returned_k: int
    latency_ms: int
    top_examples: List[Tuple[str, float]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SystemState:
    user_profile: UserProfile
    query_context: QueryContext
    resources: Resources = field(default_factory=Resources)
    fresh_papers_buffer: List[Dict[str, Any]] = field(default_factory=list)  # from Online Search agent
    recall_config: RecallConfig = field(default_factory=RecallConfig)

    # outputs
    candidates_recall: List[Candidate] = field(default_factory=list)
    recall_trace: Dict[str, RecallStats] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# -----------------------------
# Utilities
# -----------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _title_key(title: str) -> str:
    # normalized title for fuzzy-ish dedupe
    tokens = _WORD_RE.findall((title or "").lower())
    return " ".join(tokens)

def _url_hash(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    h = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    return h

def _age_days(dt: Optional[datetime], now: datetime) -> Optional[float]:
    if dt is None:
        return None
    # ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() / 86400.0


# -----------------------------
# Query plan (very lightweight)
# -----------------------------

@dataclass
class QueryPlan:
    queries: List[str]                 # lexical queries
    keywords: List[str]                # keywords for BM25 / explanation
    query_embedding: Optional[List[float]]  # for vector recall


def build_query_plan(state: SystemState) -> QueryPlan:
    """
    Daily mode: use user's preferred_keywords + (optional) generated queries.
    Interactive mode: use user_query directly.

    You can plug an LLM here to generate:
    - diversified queries
    - keyword expansions
    - query embedding (if not present)
    """
    up = state.user_profile
    qc = state.query_context

    queries: List[str] = []
    keywords: List[str] = []

    if qc.mode == "interactive" and qc.user_query:
        q = qc.user_query.strip()
        if q:
            queries.append(q)
            keywords.extend(_WORD_RE.findall(q.lower()))
    else:
        # daily: combine generated queries + preferred keywords
        for q in qc.generated_queries:
            q = (q or "").strip()
            if q:
                queries.append(q)
                keywords.extend(_WORD_RE.findall(q.lower()))
        # also fallback to preferred keywords
        keywords.extend([k.lower() for k in up.preferred_keywords if k])

        if not queries and keywords:
            # simple query string from top keywords
            queries = [" ".join(keywords[:12])]

    # dedupe while keeping order
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    queries = _dedupe([q for q in queries if q])
    keywords = _dedupe([k for k in keywords if k])

    # embedding: use profile embedding as a default for daily
    emb = up.interest_embedding
    return QueryPlan(queries=queries, keywords=keywords, query_embedding=emb)


# -----------------------------
# RecallSource interface + implementations
# -----------------------------

class RecallSource(Protocol):
    name: str
    def retrieve(self, state: SystemState, plan: QueryPlan, topk: int) -> List[Candidate]:
        ...


class VectorRecallSource:
    name = "vec"

    def retrieve(self, state: SystemState, plan: QueryPlan, topk: int) -> List[Candidate]:
        """
        TODO: Replace with your vector DB call.

        Expected behavior:
        - Use plan.query_embedding (or derive from query text) to do ANN search
        - Return Candidate list with `scores_by_source["vec"]` filled

        Below is a stub that returns empty if no vector index or no embedding.
        """
        if state.resources.vector_index is None or plan.query_embedding is None:
            return []

        # --- TODO: implement your vector index query ---
        # Example expected return from your index:
        # results = state.resources.vector_index.search(vector=plan.query_embedding, topk=topk)
        # where results items contain: id, title, abstract, authors, venue, published_at, url, score

        results = []  # placeholder

        out: List[Candidate] = []
        for r in results[:topk]:
            c = Candidate(
                paper_id=str(r["id"]),
                title=r.get("title", ""),
                abstract=r.get("abstract"),
                authors=r.get("authors"),
                venue=r.get("venue"),
                published_at=r.get("published_at"),
                url=r.get("url"),
            )
            score = _safe_float(r.get("score"), 0.0)
            c.sources.add(self.name)
            c.scores_by_source[self.name] = score
            c.reasons.append("semantic similarity to your interests")
            # lightweight features
            c.features["sim_user"] = score
            out.append(c)

        return out


class BM25RecallSource:
    name = "bm25"

    def retrieve(self, state: SystemState, plan: QueryPlan, topk: int) -> List[Candidate]:
        """
        TODO: Replace with your BM25 / lexical index call.

        Input: plan.queries / plan.keywords
        Output: Candidate list with `scores_by_source["bm25"]`
        """
        if state.resources.bm25_index is None:
            return []

        # Choose a query string
        q = plan.queries[0] if plan.queries else " ".join(plan.keywords[:12])
        if not q.strip():
            return []

        # --- TODO: implement your BM25 query ---
        # results = state.resources.bm25_index.search(query=q, topk=topk)
        results = []  # placeholder

        out: List[Candidate] = []
        for r in results[:topk]:
            c = Candidate(
                paper_id=str(r["id"]),
                title=r.get("title", ""),
                abstract=r.get("abstract"),
                authors=r.get("authors"),
                venue=r.get("venue"),
                published_at=r.get("published_at"),
                url=r.get("url"),
            )
            score = _safe_float(r.get("score"), 0.0)
            c.sources.add(self.name)
            c.scores_by_source[self.name] = score
            c.reasons.append("keyword match to your query/interests")
            c.features["keyword_overlap"] = r.get("keyword_overlap")
            out.append(c)

        return out


class FreshRecallSource:
    name = "fresh"

    def retrieve(self, state: SystemState, plan: QueryPlan, topk: int) -> List[Candidate]:
        """
        Uses state.fresh_papers_buffer (from Online Search agent).
        This source should ensure daily novelty even if indexes lag behind.

        We do a very simple relevance heuristic here:
        - score by keyword overlap with plan.keywords / query tokens
        - plus recency bonus if published_at exists
        """
        buf = state.fresh_papers_buffer or []
        if not buf:
            return []

        kw = set(plan.keywords)
        q_tokens = kw if kw else set()

        now = _now_utc()
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for item in buf:
            title = (item.get("title") or "").strip()
            abstract = item.get("abstract")
            text = (title + " " + (abstract or "")).lower()
            toks = set(_WORD_RE.findall(text))

            overlap = len(toks & q_tokens) if q_tokens else 0
            published_at = item.get("published_at")
            age = _age_days(published_at, now) if isinstance(published_at, datetime) else None
            recency_boost = 0.0
            if age is not None:
                # very small bonus: newer => larger
                recency_boost = max(0.0, 30.0 - age) / 30.0  # 0..1 if within 30 days

            # base heuristic score
            score = overlap * 1.0 + recency_boost * 2.0

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Candidate] = []
        for score, r in scored[:topk]:
            c = Candidate(
                paper_id=str(r.get("paper_id") or r.get("id") or r.get("url") or _title_key(r.get("title", ""))),
                title=r.get("title", ""),
                abstract=r.get("abstract"),
                authors=r.get("authors"),
                venue=r.get("venue"),
                published_at=r.get("published_at"),
                url=r.get("url"),
            )
            c.sources.add(self.name)
            c.scores_by_source[self.name] = float(score)
            if score > 0:
                c.reasons.append("new paper with overlap to your interests/query")
            else:
                c.reasons.append("new paper (freshness diversity)")
            c.features["recency_days"] = _age_days(c.published_at, _now_utc())
            out.append(c)

        return out


# -----------------------------
# Normalization + merge/dedupe + filtering
# -----------------------------

def normalize_scores_rank(cands: List[Candidate], source: str) -> None:
    """
    Rank-based normalization per source:
      highest score -> 1.0, lowest -> 0.0
    Robust when scores are not comparable across sources.
    """
    vals = []
    for c in cands:
        if source in c.scores_by_source:
            vals.append((_safe_float(c.scores_by_source[source]), c))
    if not vals:
        return

    # sort by raw score desc
    vals.sort(key=lambda x: x[0], reverse=True)
    n = len(vals)
    if n == 1:
        vals[0][1].scores_by_source[source] = 1.0
        return

    for rank, (raw, c) in enumerate(vals):
        # 1.0 for top, 0.0 for bottom
        norm = 1.0 - (rank / (n - 1))
        c.scores_by_source[source] = float(norm)


def merge_and_dedupe(all_lists: List[List[Candidate]]) -> List[Candidate]:
    """
    Dedupe primarily by paper_id, then url_hash, then normalized title key.
    Merges:
      - sources union
      - scores_by_source merged
      - reasons concatenated unique
      - keep best fields (prefer non-empty)
    """
    by_id: Dict[str, Candidate] = {}
    by_url: Dict[str, str] = {}     # url_hash -> paper_id (canonical)
    by_title: Dict[str, str] = {}   # title_key -> paper_id

    def merge_into(base: Candidate, incoming: Candidate) -> None:
        base.sources |= incoming.sources
        base.scores_by_source.update(incoming.scores_by_source)
        # keep richer metadata
        if (not base.abstract) and incoming.abstract:
            base.abstract = incoming.abstract
        if (not base.authors) and incoming.authors:
            base.authors = incoming.authors
        if (not base.venue) and incoming.venue:
            base.venue = incoming.venue
        if (not base.published_at) and incoming.published_at:
            base.published_at = incoming.published_at
        if (not base.url) and incoming.url:
            base.url = incoming.url

        # reasons unique
        seen = set(base.reasons)
        for r in incoming.reasons:
            if r not in seen:
                base.reasons.append(r)
                seen.add(r)

        # features: prefer existing, fill missing
        for k, v in incoming.features.items():
            if k not in base.features or base.features[k] is None:
                base.features[k] = v

    for cand_list in all_lists:
        for c in cand_list:
            pid = c.paper_id.strip() if c.paper_id else ""
            tkey = _title_key(c.title or "")
            uhash = _url_hash(c.url)

            # resolve canonical key
            canonical_id: Optional[str] = None
            if pid and pid in by_id:
                canonical_id = pid
            elif uhash and uhash in by_url:
                canonical_id = by_url[uhash]
            elif tkey and tkey in by_title:
                canonical_id = by_title[tkey]

            if canonical_id is None:
                # new canonical
                if not pid:
                    # fabricate a stable id from title+url if missing
                    pid = "gen_" + hashlib.sha256(((c.title or "") + "|" + (c.url or "")).encode("utf-8")).hexdigest()[:16]
                    c.paper_id = pid
                by_id[pid] = c
                if uhash:
                    by_url[uhash] = pid
                if tkey:
                    by_title[tkey] = pid
            else:
                merge_into(by_id[canonical_id], c)

    return list(by_id.values())


def compute_recall_score(c: Candidate, cfg: RecallConfig) -> float:
    w = cfg.weights
    s_vec = _safe_float(c.scores_by_source.get("vec"), 0.0)
    s_bm25 = _safe_float(c.scores_by_source.get("bm25"), 0.0)
    s_fresh = _safe_float(c.scores_by_source.get("fresh"), 0.0)
    s_cf = _safe_float(c.scores_by_source.get("cf"), 0.0)

    score = (
        w.get("w_vec", 0.0) * s_vec +
        w.get("w_bm25", 0.0) * s_bm25 +
        w.get("w_fresh", 0.0) * s_fresh +
        w.get("w_cf", 0.0) * s_cf
    )

    # multi-source bonus
    if len(c.sources) >= 2:
        bonus = min(w.get("bonus_multi_cap", 0.10), w.get("bonus_multi", 0.05) * (len(c.sources) - 1))
        score += bonus

    return float(score)


def light_filter(cands: List[Candidate], cfg: RecallConfig) -> List[Candidate]:
    f = cfg.filters
    now = _now_utc()
    max_age = f.get("max_age_days", None)
    min_title_len = f.get("min_title_len", 0)
    drop_missing_abs = bool(f.get("drop_if_missing_abstract", False))

    out: List[Candidate] = []
    for c in cands:
        title = (c.title or "").strip()
        if len(title) < min_title_len:
            continue
        if drop_missing_abs and not (c.abstract and c.abstract.strip()):
            continue

        if max_age is not None:
            age = _age_days(c.published_at, now)
            if age is not None and age > float(max_age):
                continue

        out.append(c)
    return out


# -----------------------------
# The Recall Node (LangGraph node function)
# -----------------------------

DEFAULT_SOURCES: List[RecallSource] = [
    VectorRecallSource(),
    BM25RecallSource(),
    FreshRecallSource(),
]

def recall_node(state: SystemState, sources: Optional[List[RecallSource]] = None) -> SystemState:
    """
    LangGraph node signature: (state) -> state
    Updates:
      - state.candidates_recall
      - state.recall_trace
    """
    cfg = state.recall_config
    srcs = sources or DEFAULT_SOURCES

    plan = build_query_plan(state)

    per_source_lists: List[List[Candidate]] = []
    trace: Dict[str, RecallStats] = {}

    # 1) Run recall sources (serial; easy to debug)
    for src in srcs:
        topk = int(cfg.per_source_topk.get(src.name, 200))
        t0 = time.perf_counter()
        try:
            cands = src.retrieve(state, plan, topk=topk)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            # ensure source tags exist
            for c in cands:
                c.sources.add(src.name)
                if src.name not in c.scores_by_source:
                    c.scores_by_source[src.name] = 0.0

            trace[src.name] = RecallStats(
                requested_k=topk,
                returned_k=len(cands),
                latency_ms=latency_ms,
                top_examples=[(c.paper_id, _safe_float(c.scores_by_source.get(src.name), 0.0)) for c in cands[:3]],
            )
            per_source_lists.append(cands)

        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            trace[src.name] = RecallStats(
                requested_k=topk,
                returned_k=0,
                latency_ms=latency_ms,
                top_examples=[],
                error=str(e),
            )
            # soft-fail: record error but continue
            state.errors.append(f"[recall:{src.name}] {e}")

    # 2) Normalize per source (rank-based)
    for src in srcs:
        # normalize each list in-place
        for lst in per_source_lists:
            normalize_scores_rank(lst, src.name)

    # 3) Merge & dedupe
    merged = merge_and_dedupe(per_source_lists)

    # 4) Aggregate recall_score
    for c in merged:
        c.recall_score = compute_recall_score(c, cfg)

        # fill some common features
        if "recency_days" not in c.features:
            c.features["recency_days"] = _age_days(c.published_at, _now_utc())

        if "author_match" not in c.features and c.authors and state.user_profile.author_affinity:
            aff = set(a.lower() for a in state.user_profile.author_affinity)
            c.features["author_match"] = int(any(a.lower() in aff for a in c.authors))

    # 5) Light filter + cap
    merged = light_filter(merged, cfg)
    merged.sort(key=lambda x: x.recall_score, reverse=True)

    final_cap = int(cfg.filters.get("final_cap", 500))
    merged = merged[:final_cap]

    # 6) Write back
    trace["merged"] = RecallStats(
        requested_k=sum(int(cfg.per_source_topk.get(s.name, 0)) for s in srcs),
        returned_k=len(merged),
        latency_ms=0,
        top_examples=[(c.paper_id, c.recall_score) for c in merged[:3]],
        error=None,
    )

    state.candidates_recall = merged
    state.recall_trace = trace
    return state


# -----------------------------
# Optional: tiny demo (safe, does not require external indexes)
# -----------------------------
if __name__ == "__main__":
    # Demo only: FreshRecallSource will work if fresh_papers_buffer is filled.
    st = SystemState(
        user_profile=UserProfile(
            user_id="u1",
            preferred_keywords=["retrieval", "reranking", "rag", "LLM"],
        ),
        query_context=QueryContext(mode="daily", generated_queries=["RAG retrieval reranking"]),
        fresh_papers_buffer=[
            {
                "paper_id": "arxiv:1234.5678",
                "title": "A Retrieval-Augmented Generation Approach for Efficient Reranking",
                "abstract": "We study RAG and reranking for LLM systems...",
                "authors": ["A. Author", "B. Author"],
                "venue": "arXiv",
                "published_at": datetime(2026, 2, 10, tzinfo=timezone.utc),
                "url": "https://arxiv.org/abs/1234.5678",
            },
            {
                "paper_id": "arxiv:9999.0001",
                "title": "Unrelated Topic in Vision",
                "abstract": "This paper is about segmentation...",
                "authors": ["C. Author"],
                "venue": "arXiv",
                "published_at": datetime(2026, 2, 11, tzinfo=timezone.utc),
                "url": "https://arxiv.org/abs/9999.0001",
            },
        ],
        resources=Resources(vector_index=None, bm25_index=None),
    )

    st = recall_node(st)
    print("Recall candidates:", len(st.candidates_recall))
    for c in st.candidates_recall[:5]:
        print(c.paper_id, c.recall_score, sorted(list(c.sources)), c.title)
    print("Trace:", {k: asdict(v) for k, v in st.recall_trace.items()})
