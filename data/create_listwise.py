# -*- coding: utf-8 -*-
"""
Listwise data generation script (RAG + paper API + weighted scoring).

1. Seed papers: from RAG vector DB. For each seed, get reference list via paper API (OpenAlex).
2. Candidates: 2 positive (from citation list), 4 negative (from RAG by similarity percentiles).
   - Positives: resolve from RAG first; if <2 in RAG, fetch from API and add to RAG.
   - Negatives: 3 from similarity top 10%-40% (not cited), 1 from similarity 70%+ (not cited).
3. Label = weighted: 0.6 * cited + 0.3 * cosine_sim + 0.1 * llm_score_norm (weights configurable).
4. Each sample includes current_paper: {id, title, abstract} (the seed paper being viewed).
5. Default output: /tmp/listwise_data/listwise_output.json. Mode (test/formal) unchanged.
6. 支持多进程并行 (-j)，因含 LLM 调用建议 -j 4 左右，避免 API 限流。
"""
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Stub posthog before ChromaDB loads; posthog uses Python 3.9+ type hints (dict[str, ...])
# that raise "'type' object is not subscriptable" on Python 3.8.
class _PosthogStub:
    disabled = True
    project_api_key = ""

    @staticmethod
    def capture(*args, **kwargs):
        pass


sys.modules["posthog"] = _PosthogStub()

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from listwise_prompts import build_profile_query_prompt, build_score_candidates_prompt

# RAG + paper API helpers
try:
    from rag_listwise_helpers import (
        get_seed_paper_ids_from_rag,
        get_paper_from_rag,
        get_papers_from_rag,
        get_embedding_for_paper,
        similarity_search_rag,
        add_paper_to_rag,
        cosine_similarity,
    )
except ImportError:
    get_seed_paper_ids_from_rag = None
    get_paper_from_rag = None
    get_papers_from_rag = None
    get_embedding_for_paper = None
    similarity_search_rag = None
    add_paper_to_rag = None
    cosine_similarity = lambda a, b: 0.0

# -----------------------------------------------------------------------------
# Env and LLM config
# -----------------------------------------------------------------------------
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
QWEN_BASE = (os.getenv("QWEN_API_BASE") or os.getenv("DASHSCOPE_API_BASE") or "").rstrip("/")
if not QWEN_BASE:
    QWEN_BASE = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = os.getenv("QWEN_MODEL") or "qwen-plus"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = (os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_BASE = "https://api.openalex.org"

# Candidates: 2 positive (from citation list), 4 negative (from RAG)
NUM_POSITIVE = 2
NUM_NEGATIVE_HARD = 3   # similarity in top 10%-40%, not cited
NUM_NEGATIVE_EASY = 1   # similarity 70%+ (排名 70% 往后), not cited
TOTAL_CANDIDATES = NUM_POSITIVE + NUM_NEGATIVE_HARD + NUM_NEGATIVE_EASY  # 6

# Label weights: cited | cosine_sim | llm_norm
WEIGHT_CITED = 0.6
WEIGHT_SIM = 0.3
WEIGHT_LLM = 0.1

# Default sample counts (test vs formal)
NUM_SAMPLES = 5
TEST_NUM_SAMPLES = 2

# Default output path
DEFAULT_OUTPUT_PATH = "/tmp/listwise_data/listwise_output.json"


# -----------------------------------------------------------------------------
# Parse LLM JSON output
# -----------------------------------------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    # Strip markdown code blocks
    for pattern in [r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        m = re.search(pattern, text)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_profile_query_response(response_text: str) -> Tuple[Optional[dict], Optional[str]]:
    obj = _extract_json(response_text)
    if not obj:
        return None, None
    profile = obj.get("user_profile")
    query = obj.get("query")
    if isinstance(query, str):
        query = query.strip()
    return profile, query


def _parse_scores_response(response_text: str, n: int) -> Optional[List[int]]:
    obj = _extract_json(response_text)
    if not obj:
        return None
    scores = obj.get("scores")
    if not isinstance(scores, list) or len(scores) != n:
        return None
    out = []
    for s in scores:
        try:
            v = int(s)
            if v not in (0, 1, 2):
                v = max(0, min(2, v))
            out.append(v)
        except (TypeError, ValueError):
            out.append(0)
    return out if len(out) == n else None


def _llm_scores_to_norm(scores: List[int]) -> List[float]:
    """Convert 0/1/2 LLM scores to 0-1 scale (divide by 2)."""
    return [max(0.0, min(1.0, s / 2.0)) for s in scores]


# -----------------------------------------------------------------------------
# OpenAlex API
# -----------------------------------------------------------------------------
def _openalex_headers():
    h = {}
    if OPENALEX_EMAIL:
        h["User-Agent"] = f"mailto:{OPENALEX_EMAIL}"
    return h


def get_reference_list_from_s2(arxiv_id: str) -> List[str]:
    """Get reference list from Semantic Scholar when OpenAlex fails. Returns arxiv IDs when available."""
    arxiv_id = (arxiv_id or "").strip()
    if not _is_arxiv_id(arxiv_id):
        return []
    s2_id = f"ARXIV:{arxiv_id}" if not arxiv_id.upper().startswith("ARXIV:") else arxiv_id
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}/references",
            params={"fields": "citedPaper.paperId,citedPaper.externalIds", "limit": 50},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        ids = []
        for item in data.get("data", []):
            cp = item.get("citedPaper") or {}
            ext = cp.get("externalIds") or {}
            aid = ext.get("ArXiv")
            if aid:
                ids.append(aid)
            elif cp.get("paperId"):
                ids.append(cp["paperId"])  # S2 paperId 作为备选
        return ids
    except Exception:
        return []


def get_reference_list_by_paper_id(paper_id: str) -> List[str]:
    """Get reference list (cited paper IDs). OpenAlex first, fallback to Semantic Scholar for arxiv IDs."""
    raw_id = paper_id
    if raw_id.startswith("http"):
        raw_id = raw_id.rstrip("/").split("/")[-1]
    if raw_id.upper().startswith("W"):
        raw_id = raw_id[1:]

    w = get_work(paper_id)
    if w:
        refs = w.get("referenced_works") or []
        ids = []
        for r in refs:
            if isinstance(r, str):
                wid = r.rstrip("/").split("/")[-1]
                if wid and wid not in ids:
                    ids.append(wid)
        return ids

    # OpenAlex 失败时，arxiv 用 S2
    if _is_arxiv_id(raw_id):
        return get_reference_list_from_s2(raw_id)
    return []


def _is_arxiv_id(wid: str) -> bool:
    """Check if id looks like arxiv (e.g. 1908.02905 or 2101.12345)."""
    s = (wid or "").strip()
    if s.upper().startswith("W"):
        s = s[1:]
    return bool(re.match(r"^\d{4}\.\d{4,5}$", s))


def get_work(work_id: str) -> Optional[dict]:
    """Get a single work by id (OpenAlex W123, arxiv 1908.02905, or URL)."""
    wid = work_id
    if wid.startswith("http"):
        wid = wid.rstrip("/").split("/")[-1]
    if wid.upper().startswith("W"):
        wid = wid[1:]

    # arxiv ID: OpenAlex 部分版本支持 /works/arxiv:xx，失败则返回 None
    if _is_arxiv_id(wid):
        try:
            r = requests.get(
                f"{OPENALEX_BASE}/works/arxiv:{wid}",
                headers=_openalex_headers(),
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            # arxiv 查不到时静默返回 None，不打印（避免刷屏）
            return None

    url = f"{OPENALEX_BASE}/works/W{wid}"
    try:
        r = requests.get(url, headers=_openalex_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"OpenAlex get_work {work_id} error: {e}")
        return None


def get_abstract_from_s2(doi: str = None, title: str = None) -> str:
    """Get abstract from Semantic Scholar (no API key needed for abstract field)."""
    if not doi and not title:
        return ""
    
    # If we have DOI, try direct lookup first (more reliable)
    if doi:
        # Clean DOI format
        doi_clean = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
        try:
            # Try direct DOI lookup
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_clean}"
            r = requests.get(url, params={"fields": "abstract"}, timeout=10)
            if r.status_code == 200:
                paper = r.json()
                abstract = paper.get("abstract", "")
                if abstract and len(abstract.strip()) >= 50:
                    return abstract
        except:
            pass
    
    # Fallback to search API (for DOI that didn't work or title-only)
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    query = doi if doi else title
    params = {
        "query": query,
        "limit": 3,  # Get a few results to find best match
        "fields": "abstract,title,paperId"
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data", [])
        if papers:
            # If we have DOI, try to find exact match
            if doi:
                doi_clean = doi.replace("https://doi.org/", "").replace("doi:", "").strip().lower()
                for paper in papers:
                    paper_id = paper.get("paperId", "").lower()
                    # S2 paperId might contain DOI or we can check title similarity
                    if doi_clean in paper_id or paper_id in doi_clean:
                        abstract = paper.get("abstract", "")
                        if abstract and len(abstract.strip()) >= 50:
                            return abstract
            
            # Use first result (best match from search)
            abstract = papers[0].get("abstract", "")
            if abstract and len(abstract.strip()) >= 50:
                return abstract
    except Exception:
        # Silent fail - return empty
        pass
    return ""


def work_to_candidate(w: dict) -> dict:
    """Convert OpenAlex work to candidate dict (title, abstract, authors, paper_id)."""
    title = (w.get("title") or w.get("display_name") or "").strip()
    abstract = ""
    
    # Try to reconstruct from inverted index first
    if w.get("abstract_inverted_index") and isinstance(w.get("abstract_inverted_index"), dict):
        pairs = []
        for word, positions in w["abstract_inverted_index"].items():
            for p in positions:
                pairs.append((p, word))
        pairs.sort(key=lambda x: x[0])
        abstract = " ".join(p[1] for p in pairs)[:500]
    
    # If no abstract or too short (< 50 chars), try Semantic Scholar
    if not abstract or len(abstract.strip()) < 50:
        doi = w.get("doi", "")
        if doi:
            # Clean DOI format
            if isinstance(doi, str):
                doi = doi.replace("https://doi.org/", "").replace("doi:", "").strip()
            s2_abstract = get_abstract_from_s2(doi=doi, title=title)
            if s2_abstract and len(s2_abstract.strip()) >= 50:
                abstract = s2_abstract
        
        # If still no good abstract, try with title only
        if not abstract or len(abstract.strip()) < 50:
            s2_abstract = get_abstract_from_s2(title=title)
            if s2_abstract and len(s2_abstract.strip()) >= 50:
                abstract = s2_abstract
        
        # Final fallback
        if not abstract or len(abstract.strip()) < 50:
            abstract = "(Abstract not available)"
    
    authors = []
    for a in w.get("authorships") or []:
        author = a.get("author") or {}
        name = author.get("display_name") or a.get("raw_author_name")
        if name:
            authors.append(name)
    paper_id = (w.get("id") or "").split("/")[-1] or w.get("id")
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors[:10],
    }


def get_author_info_and_field(work: dict) -> Tuple[str, str]:
    """Extract author info string and field/topic string from work."""
    author_parts = []
    for a in work.get("authorships") or []:
        author = a.get("author") or {}
        name = author.get("display_name") or a.get("raw_author_name") or "Unknown"
        pos = a.get("author_position", "")
        insts = [i.get("display_name") for i in (a.get("institutions") or []) if i.get("display_name")]
        inst_str = ", ".join(insts[:2]) if insts else ""
        line = f"  - {name}"
        if pos:
            line += f" (position: {pos})"
        if inst_str:
            line += f", institution: {inst_str}"
        author_parts.append(line)
    author_info = "\n".join(author_parts) if author_parts else "  (none)"

    topic_parts = []
    pt = work.get("primary_topic") or {}
    if pt.get("display_name"):
        topic_parts.append(pt["display_name"])
    for t in work.get("topics") or []:
        if t.get("display_name") and t.get("display_name") not in topic_parts:
            topic_parts.append(t["display_name"])
    for c in work.get("concepts") or []:
        if c.get("display_name") and c.get("display_name") not in topic_parts:
            topic_parts.append(c["display_name"])
    field_topic = ", ".join(topic_parts[:10]) if topic_parts else "(none)"

    return author_info, field_topic


def extract_paper_tags(work: dict) -> List[str]:
    """Extract a list of tags from the current paper (topics/concepts/keywords)."""
    tags: List[str] = []

    def _add(x: Optional[str]):
        x = (x or "").strip()
        if not x:
            return
        if x not in tags:
            tags.append(x)

    pt = work.get("primary_topic") or {}
    _add(pt.get("display_name"))

    for t in work.get("topics") or []:
        _add(t.get("display_name"))
        sub = (t.get("subfield") or {}).get("display_name")
        fld = (t.get("field") or {}).get("display_name")
        dom = (t.get("domain") or {}).get("display_name")
        _add(sub)
        _add(fld)
        _add(dom)

    for c in work.get("concepts") or []:
        _add(c.get("display_name"))

    for k in work.get("keywords") or []:
        # keyword objects usually have display_name
        if isinstance(k, dict):
            _add(k.get("display_name") or k.get("keyword"))
        elif isinstance(k, str):
            _add(k)

    # Keep it compact and stable
    return tags[:30]


def is_target_area_paper(work: dict) -> bool:
    """Heuristic filter: keep papers likely in NLP/CV/LLM/RAG/Agent areas."""
    tags = extract_paper_tags(work)
    hay = " | ".join(tags).lower()
    keywords = [
        "natural language", "nlp", "language model", "large language", "llm",
        "computer vision", "vision", "image", "multimodal",
        "retrieval", "rag", "retrieval-augmented", "retrieval augmented",
        "agent", "tool use", "planning", "autonomous",
    ]
    return any(k in hay for k in keywords)


def normalize_interest_fields(user_profile: dict, available_tags: List[str]) -> dict:
    """Ensure interest_field is a list of 3-5 items, all chosen from available_tags."""
    profile = dict(user_profile or {})
    chosen = profile.get("interest_field")
    avail_set = {t.lower(): t for t in (available_tags or [])}

    fields: List[str] = []
    if isinstance(chosen, list):
        for x in chosen:
            s = str(x).strip()
            if not s:
                continue
            # map by lowercase exact match if possible
            s2 = avail_set.get(s.lower(), s)
            if s2 in available_tags and s2 not in fields:
                fields.append(s2)
    elif isinstance(chosen, str) and chosen.strip():
        # allow comma-separated string
        parts = [p.strip() for p in re.split(r"[;,/|]", chosen) if p.strip()]
        for p in parts:
            s2 = avail_set.get(p.lower(), p)
            if s2 in available_tags and s2 not in fields:
                fields.append(s2)

    # fallback: pick from available tags
    if len(fields) < 3:
        for t in available_tags:
            if t not in fields:
                fields.append(t)
            if len(fields) >= 3:
                break
    fields = fields[:5]
    profile["interest_field"] = fields
    return profile


def get_seed_paper_info(seed_id: str) -> Tuple[Optional[Dict], Optional[dict]]:
    """Get seed paper info for current_paper and profile. Returns (current_paper_dict, openalex_work_or_none)."""
    # 原始 ID（metadata_db / Chroma 存的是 arxiv 格式如 1908.02905）
    raw_id = seed_id
    if raw_id.startswith("http"):
        raw_id = raw_id.rstrip("/").split("/")[-1]
    if raw_id.upper().startswith("W"):
        raw_id = raw_id[1:]

    # Try RAG/metadata_db first（用原始 arxiv id 查，不用 W 前缀）
    rag_paper = get_paper_from_rag(raw_id) if get_paper_from_rag else None
    if rag_paper:
        current = {"id": raw_id, "title": rag_paper.get("title", ""), "abstract": rag_paper.get("abstract", "")}
        # Optionally fetch OpenAlex work for author/field/tags (profile generation)
        work = get_work(raw_id)
        return current, work

    # Fetch from OpenAlex
    w = get_work(raw_id)
    if not w:
        return None, None
    title = w.get("title") or w.get("display_name") or ""
    abstract = ""
    if w.get("abstract_inverted_index"):
        pairs = []
        for word, positions in (w.get("abstract_inverted_index") or {}).items():
            for p in positions:
                pairs.append((p, word))
        pairs.sort(key=lambda x: x[0])
        abstract = " ".join(p[1] for p in pairs)[:500]
    if not abstract:
        abstract = get_abstract_from_s2(doi=w.get("doi"), title=title) or "(No abstract)"
    current = {"id": raw_id, "title": title, "abstract": abstract}
    return current, w


def get_paper_from_s2(paper_id: str) -> Optional[Dict[str, Any]]:
    """Fetch paper from Semantic Scholar by arxiv id or S2 paperId. Returns {paper_id, title, abstract, authors}."""
    pid = (paper_id or "").strip()
    if not pid:
        return None
    # S2 接受 ARXIV:xxx 或 corpusId
    if _is_arxiv_id(pid):
        s2_id = f"ARXIV:{pid}"
    elif not pid.startswith("ARXIV:") and len(pid) == 40:  # S2 paperId
        s2_id = pid
    else:
        return None
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}",
            params={"fields": "title,abstract,authors,externalIds"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        title = (data.get("title") or "").strip()
        abstract = (data.get("abstract") or "").strip()
        if not abstract and data.get("externalIds", {}).get("ArXiv"):
            abstract = get_abstract_from_s2(title=title) or "(Abstract not available)"
        authors = [a.get("name", "") for a in (data.get("authors") or [])[:10] if a.get("name")]
        aid = data.get("externalIds", {}).get("ArXiv") or pid
        return {"paper_id": aid, "title": title, "abstract": abstract or "(Abstract not available)", "authors": authors}
    except Exception:
        return None


def resolve_positive_candidate(ref_id: str) -> Optional[Dict[str, Any]]:
    """Resolve one positive candidate from ref list: RAG first, else OpenAlex, else S2."""
    rid = ref_id
    if rid.startswith("http"):
        rid = rid.rstrip("/").split("/")[-1]
    # 查 RAG 时用原始 id（arxiv 存的是 1908.02905 格式，无 W 前缀）
    lookup_id = rid[1:] if rid.upper().startswith("W") and _is_arxiv_id(rid[1:]) else rid

    p = get_paper_from_rag(lookup_id) if get_paper_from_rag else None
    if p:
        return {"paper_id": lookup_id, "title": p.get("title", ""), "abstract": p.get("abstract", ""), "authors": p.get("authors", [])}

    w = get_work(rid)
    if w:
        c = work_to_candidate(w)
        if add_paper_to_rag and c:
            add_paper_to_rag(c)
        return c

    # OpenAlex 失败时，arxiv 或 S2 paperId 用 S2
    c = get_paper_from_s2(rid)
    if c and add_paper_to_rag:
        add_paper_to_rag(c)
    return c


def get_negative_candidates_from_rag(
    seed_embedding: List[float],
    seed_id: str,
    ref_ids: List[str],
    num_hard: int = NUM_NEGATIVE_HARD,
    num_easy: int = NUM_NEGATIVE_EASY,
) -> List[Tuple[Dict[str, Any], float]]:
    """Get negative candidates from RAG: num_hard from similarity 10%-40%, num_easy from 70%+ (排名 70% 往后). Returns [(candidate_dict, sim_score), ...]."""
    exclude = set([seed_id] + list(ref_ids))
    raw = similarity_search_rag(seed_embedding, top_k=500, exclude_ids=list(exclude)) if similarity_search_rag else []
    if not raw:
        return []

    # raw is sorted by score desc (higher = more similar)
    n = len(raw)
    if n == 0:
        return []
    idx_10 = int(0.10 * n)
    idx_30 = int(0.30 * n)
    idx_85 = int(0.85 * n)
    # 10%-30% band: indices [idx_10, idx_30)
    band_hard = raw[idx_10:idx_30] if idx_30 > idx_10 else raw[:1]
    # 85% 往后: indices [idx_85, end]
    band_easy = raw[idx_85:] if idx_85 < n else []

    out = []
    for i, r in enumerate(band_hard[:num_hard]):
        pid = r.get("paper_id")
        score = float(r.get("score", 0))
        p = get_paper_from_rag(pid) if get_paper_from_rag else None
        if p:
            out.append(({"paper_id": pid, "title": p.get("title", ""), "abstract": p.get("abstract", ""), "authors": p.get("authors", [])}, score))
    for r in band_easy[:num_easy]:
        pid = r.get("paper_id")
        score = float(r.get("score", 0))
        p = get_paper_from_rag(pid) if get_paper_from_rag else None
        if p:
            out.append(({"paper_id": pid, "title": p.get("title", ""), "abstract": p.get("abstract", ""), "authors": p.get("authors", [])}, score))
        if len(out) >= num_hard + num_easy:
            break
    return out


# -----------------------------------------------------------------------------
# LLM calls: Qwen / DeepSeek
# -----------------------------------------------------------------------------
def _chat_openai_compatible(api_key: str, base_url: str, model: str, messages: List[dict], temperature: float = 0.3) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    client = OpenAI(api_key=api_key, base_url=base_url)
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return (r.choices[0].message.content or "").strip()


def generate_profile_and_query(
    title: str,
    abstract: str,
    author_info: str,
    field_topic: str,
    available_tags: List[str],
) -> Tuple[Optional[dict], Optional[str]]:
    """Call Qwen to generate user_profile and query."""
    if not QWEN_API_KEY:
        print("QWEN_API_KEY / DASHSCOPE_API_KEY not set; skipping profile and query generation")
        return None, None
    messages = build_profile_query_prompt(title, abstract, author_info, field_topic, available_tags=available_tags)
    try:
        text = _chat_openai_compatible(QWEN_API_KEY, QWEN_BASE, QWEN_MODEL, messages, temperature=0.5)
        return _parse_profile_query_response(text)
    except Exception as e:
        print(f"Qwen generate_profile_and_query error: {e}")
        return None, None


def score_candidates_with_deepseek(
    user_profile: dict,
    query: str,
    candidates: List[dict],
) -> List[int]:
    """Score candidates once with DeepSeek; return list of 0/1/2 of length len(candidates)."""
    n = len(candidates)
    if not DEEPSEEK_API_KEY:
        return [0] * n
    messages = build_score_candidates_prompt(user_profile, query, candidates, max_abstract_chars=300)
    try:
        text = _chat_openai_compatible(DEEPSEEK_API_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL, messages, temperature=0.2)
        s = _parse_scores_response(text, n)
        if s:
            return [max(0, min(2, int(round(x)))) for x in s]
    except Exception as e:
        print(f"DeepSeek score error: {e}")
    return [0] * n


# -----------------------------------------------------------------------------
# Main: build one listwise sample (RAG + 2 pos / 4 neg + weighted label)
# -----------------------------------------------------------------------------
def build_one_listwise_sample(seed_id: str, mode: str = "formal") -> Optional[dict]:
    """Build one listwise sample: seed from RAG, ref list from API, 2 pos + 4 neg, weighted label."""
    # 1) Seed paper info (current_paper + author/tags for profile/query)
    current_paper, work = get_seed_paper_info(seed_id)
    if not current_paper:
        return None
    title = current_paper.get("title", "")
    abstract = current_paper.get("abstract", "")
    if not work:
        author_info = "  (none)"
        field_topic = "(none)"
        available_tags = ["machine learning", "nlp", "llm"]
    else:
        author_info, field_topic = get_author_info_and_field(work)
        available_tags = extract_paper_tags(work)

    user_profile, query = generate_profile_and_query(title, abstract, author_info, field_topic, available_tags)
    if not user_profile:
        user_profile = {"position": "researcher", "goal": "find related work", "professional_level": "intermediate", "interest_field": available_tags[:3] if available_tags else ["machine learning", "nlp", "llm"]}
    user_profile = normalize_interest_fields(user_profile, available_tags)
    if not query:
        query = f"Recommend papers similar to \"{title[:50]}\" or in the same field"

    canonical_id = current_paper.get("id") or seed_id
    # 2) Reference list from API
    ref_ids = get_reference_list_by_paper_id(canonical_id)
    if len(ref_ids) < NUM_POSITIVE:
        return None

    # 3) Two positives from ref list (RAG first, else API and add to RAG)
    pos_candidates = []
    for rid in ref_ids[:NUM_POSITIVE * 2]:
        c = resolve_positive_candidate(rid)
        if c:
            pos_candidates.append(c)
            if len(pos_candidates) >= NUM_POSITIVE:
                break
        time.sleep(0.1)
    if len(pos_candidates) < NUM_POSITIVE:
        return None
    pos_candidates = pos_candidates[:NUM_POSITIVE]
    ref_set = set(ref_ids)

    # 4) Seed embedding and four negatives from RAG (percentiles)
    seed_emb = get_embedding_for_paper(canonical_id) if get_embedding_for_paper else None
    if not seed_emb:
        return None
    neg_with_sim = get_negative_candidates_from_rag(seed_emb, canonical_id, ref_ids, num_hard=NUM_NEGATIVE_HARD, num_easy=NUM_NEGATIVE_EASY)
    if len(neg_with_sim) < NUM_NEGATIVE_HARD + NUM_NEGATIVE_EASY:
        return None
    neg_candidates = [c for c, _ in neg_with_sim]
    neg_sims = [s for _, s in neg_with_sim]

    # 5) Build candidate list: [pos1, pos2, neg1, neg2, neg3, neg4]; each has paper_id, title, abstract, authors
    candidates = pos_candidates + neg_candidates
    # Cited and sim for each candidate
    cited_flags = [1] * len(pos_candidates) + [0] * len(neg_candidates)
    sim_scores = []
    for i, cand in enumerate(candidates):
        if i < len(pos_candidates):
            emb = get_embedding_for_paper(cand.get("paper_id")) if get_embedding_for_paper else None
            sim = cosine_similarity(seed_emb, emb) if emb else 0.0
            sim_scores.append(max(0.0, min(1.0, sim)))
        else:
            j = i - len(pos_candidates)
            sim_scores.append(neg_sims[j] if j < len(neg_sims) else 0.0)

    # 6) LLM scores (0/1/2) for weighted part（只调用 DeepSeek 一次）
    llm_scores = score_candidates_with_deepseek(user_profile, query, candidates)
    if len(llm_scores) != len(candidates):
        llm_scores = llm_scores + [0] * (len(candidates) - len(llm_scores))
    llm_norms = _llm_scores_to_norm(llm_scores[:len(candidates)])

    # 7) Weighted label per candidate
    labels = []
    for i in range(len(candidates)):
        lab = WEIGHT_CITED * cited_flags[i] + WEIGHT_SIM * sim_scores[i] + WEIGHT_LLM * (llm_norms[i] if i < len(llm_norms) else 0)
        labels.append(round(lab, 4))

    return {
        "current_paper": {"id": current_paper.get("id"), "title": current_paper.get("title"), "abstract": current_paper.get("abstract")},
        "user_profile": user_profile,
        "query": query,
        "candidate_list": candidates,
        "label": labels,
    }


def _worker_build_samples(args_tuple: Tuple[List[str], str, int]) -> List[dict]:
    """Worker: 对一批 seed_ids 生成 samples。返回 sample 列表。"""
    seed_ids, mode, worker_id = args_tuple
    samples = []
    for sid in seed_ids:
        one = build_one_listwise_sample(sid, mode=mode)
        if one:
            samples.append(one)
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Listwise data: RAG seeds + 2 pos / 4 neg + weighted label (0.6*cited+0.3*sim+0.1*llm), 多进程"
    )
    parser.add_argument("--mode", type=str, choices=["test", "formal"], default="test", help="test=quick; formal=full scoring")
    parser.add_argument("-n", "--num", type=int, default=None, help="Number of samples (test default 2, formal default 5)")
    parser.add_argument("-o", "--out", type=str, default=None, help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="并行进程数 (default: 4，因含 LLM 调用建议勿过大)")
    parser.add_argument("--openalex-email", type=str, default="", help="OpenAlex email (optional)")
    args = parser.parse_args()

    if args.num is None:
        args.num = TEST_NUM_SAMPLES if args.mode == "test" else NUM_SAMPLES
    output_path = args.out if args.out else DEFAULT_OUTPUT_PATH
    n_jobs = max(1, args.jobs)

    global OPENALEX_EMAIL
    if args.openalex_email:
        OPENALEX_EMAIL = args.openalex_email

    print(f"[mode={args.mode}] Generating {args.num} samples, {n_jobs} workers, label=0.6*cited+0.3*sim+0.1*llm")
    seed_ids = []
    if get_seed_paper_ids_from_rag:
        seed_ids = get_seed_paper_ids_from_rag(limit=max(args.num + 500, 2000))
    if not seed_ids:
        print("No seed paper IDs from RAG. Ensure vector DB is populated and backend is available.")
        return

    # 分片给各 worker
    chunk_size = max(1, (len(seed_ids) + n_jobs - 1) // n_jobs)
    chunks = [
        (seed_ids[i : i + chunk_size], args.mode, wid)
        for wid, i in enumerate(range(0, len(seed_ids), chunk_size))
    ]
    chunks = chunks[:n_jobs]

    all_samples = []
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(_worker_build_samples, chunks)
        for r in results:
            all_samples.extend(r)
            if len(all_samples) >= args.num:
                break

    samples = all_samples[: args.num]
    out_obj = {"mode": args.mode, "num_samples": len(samples), "samples": samples}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
