# -*- coding: utf-8 -*-
"""
Listwise data generation script - 多进程并行版，无 LLM 打分。

1. Seed papers: from RAG vector DB.
2. Candidates: 2 positive (citation), 4 negative (RAG similarity percentiles or --refs-only 下从其他 seed 引用池采样).
3. Label = 0.7 * cited + 0.3 * cosine_sim（无 LLM 打分）
4. 多进程并行生成，加速数据构建。
5. --refs-only: 负例仅从其他 seed 的引用列表采样，不用 RAG 相似度检索。
"""
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
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

# Local cache (optional, set via --cache-dir)
_paper_cache = None

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
CROSSREF_BASE = "https://api.crossref.org"

# Candidates: 2 positive (from citation list), 4 negative (from RAG)
NUM_POSITIVE = 2
NUM_NEGATIVE_HARD = 3   # similarity in top 10%-40%, not cited
NUM_NEGATIVE_EASY = 1   # similarity 70%+ (排名 70% 往后), not cited
TOTAL_CANDIDATES = NUM_POSITIVE + NUM_NEGATIVE_HARD + NUM_NEGATIVE_EASY  # 6

# Label weights: cited 0.7, cosine_sim 0.3（无 LLM）
WEIGHT_CITED = 0.7
WEIGHT_SIM = 0.3

# Default sample counts (test vs formal)
NUM_SAMPLES = 5
TEST_NUM_SAMPLES = 2

# Default output path
DEFAULT_OUTPUT_PATH = "/tmp/listwise_data/listwise_3.json"


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
# OpenAlex API（限速 + 429 重试，避免 Too Many Requests）
# -----------------------------------------------------------------------------
_OPENALEX_LAST_REQUEST_TIME = 0.0
_OPENALEX_MIN_GAP = 0.2  # 单进程内最少间隔秒数（多进程时由 worker 自动调大）
_GET_WORK_CACHE: Dict[str, Optional[dict]] = {}  # per-process LRU cache for get_work


def _openalex_headers():
    h = {}
    if OPENALEX_EMAIL:
        h["User-Agent"] = f"mailto:{OPENALEX_EMAIL}"
    return h


def _openalex_get(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """OpenAlex 请求：限速 + 429 时指数退避重试。"""
    global _OPENALEX_LAST_REQUEST_TIME
    for attempt in range(max_retries + 1):
        # 限速：距离上次请求至少 _OPENALEX_MIN_GAP 秒
        elapsed = time.monotonic() - _OPENALEX_LAST_REQUEST_TIME
        if elapsed < _OPENALEX_MIN_GAP:
            time.sleep(_OPENALEX_MIN_GAP - elapsed)
        _OPENALEX_LAST_REQUEST_TIME = time.monotonic()

        try:
            r = requests.get(url, headers=_openalex_headers(), timeout=15)
            if r.status_code == 429:
                backoff = 2 ** (attempt + 1)
                if attempt < max_retries:
                    time.sleep(backoff)
                    continue
            return r
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Crossref API（作为 OpenAlex 的 fallback，减轻限流压力）
# -----------------------------------------------------------------------------
_CROSSREF_LAST_REQUEST_TIME = 0.0
_CROSSREF_MIN_GAP = 0.2


def _is_doi(s: str) -> bool:
    """Check if string looks like a DOI (e.g. 10.1038/nature12373)."""
    s = (s or "").strip()
    s = s.replace("https://doi.org/", "").replace("doi:", "").strip()
    return bool(s and s.startswith("10.") and "/" in s)


def _crossref_get(url: str) -> Optional[requests.Response]:
    """Crossref 请求，带限速。Polite pool 建议设置 User-Agent 为 mailto:xxx"""
    global _CROSSREF_LAST_REQUEST_TIME
    elapsed = time.monotonic() - _CROSSREF_LAST_REQUEST_TIME
    if elapsed < _CROSSREF_MIN_GAP:
        time.sleep(_CROSSREF_MIN_GAP - elapsed)
    _CROSSREF_LAST_REQUEST_TIME = time.monotonic()
    ua = f"mailto:{OPENALEX_EMAIL}" if OPENALEX_EMAIL else "PaperRecBot/1.0"
    try:
        r = requests.get(url, headers={"User-Agent": ua}, timeout=15)
        return r
    except Exception:
        return None


def crossref_get_work_by_doi(doi: str) -> Optional[dict]:
    """Fetch work from Crossref by DOI. Returns OpenAlex-like dict for work_to_candidate."""
    doi_clean = (doi or "").replace("https://doi.org/", "").replace("doi:", "").strip()
    if not doi_clean or not doi_clean.startswith("10."):
        return None
    url = f"{CROSSREF_BASE}/works/{quote(doi_clean, safe='')}"
    r = _crossref_get(url)
    if not r or r.status_code != 200:
        return None
    try:
        data = r.json()
        msg = data.get("message") or {}
        # 转为 OpenAlex 兼容格式
        title = (msg.get("title") or [""])[0] or ""
        authors = []
        for a in msg.get("author") or []:
            given = a.get("given", "")
            family = a.get("family", "")
            name = f"{given} {family}".strip() or family
            if name:
                authors.append(name)
        abstract = ""
        ab = msg.get("abstract", "")
        if isinstance(ab, str) and len(ab.strip()) >= 50:
            abstract = ab.strip()
        elif not abstract and msg.get("DOI"):
            abstract = get_abstract_from_s2(doi=msg["DOI"], title=title) or ""
        if not abstract or len(abstract.strip()) < 50:
            abstract = "(Abstract not available)"
        refs = []
        for ref in msg.get("reference") or []:
            rdoi = ref.get("DOI")
            if rdoi and rdoi not in refs:
                refs.append(rdoi)
        return {
            "id": f"https://doi.org/{doi_clean}",
            "doi": f"https://doi.org/{msg.get('DOI', doi_clean)}",
            "paper_id": doi_clean,
            "title": title,
            "display_name": title,
            "abstract_inverted_index": None,
            "authorships": [{"author": {"display_name": n}, "raw_author_name": n} for n in authors],
            "referenced_works": refs,
        }
    except Exception:
        return None


def crossref_get_reference_list(doi: str) -> List[str]:
    """Get reference list (DOIs) from Crossref. Returns list of DOI strings."""
    w = crossref_get_work_by_doi(doi)
    if w:
        return w.get("referenced_works") or []
    return []


def crossref_query_by_arxiv(arxiv_id: str) -> Optional[dict]:
    """尝试通过 arxiv ID 在 Crossref 中查找（实验性，用于 OpenAlex 限流时的 fallback）。"""
    if not _is_arxiv_id(arxiv_id):
        return None
    url = f"{CROSSREF_BASE}/works?query.bibliographic={arxiv_id}&rows=1"
    r = _crossref_get(url)
    if not r or r.status_code != 200:
        return None
    try:
        data = r.json()
        items = data.get("message", {}).get("items") or []
        if not items:
            return None
        item = items[0]
        doi = item.get("DOI", "")
        if not doi:
            return None
        return crossref_get_work_by_doi(doi)
    except Exception:
        return None


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


def _normalize_paper_id(paper_id: str) -> str:
    """Extract clean ID (arxiv, OpenAlex wid, or DOI) from URL or prefixed form."""
    s = (paper_id or "").strip()
    if s.startswith("https://doi.org/") or s.startswith("http://doi.org/"):
        return s.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if s.startswith("http"):
        return s.rstrip("/").split("/")[-1]
    if s.upper().startswith("W") and not _is_arxiv_id(s[1:]):
        return s[1:]
    return s


def get_reference_list_by_paper_id(paper_id: str) -> List[str]:
    """Get reference list (cited paper IDs). 优先本地缓存，否则 API。"""
    raw_id = _normalize_paper_id(paper_id)

    # 本地缓存
    if _paper_cache:
        refs = _paper_cache.get_refs(raw_id)
        if refs is not None:
            return refs

    # DOI：优先 Crossref，减轻 OpenAlex 压力
    if _is_doi(raw_id):
        refs = crossref_get_reference_list(raw_id)
        if refs:
            return refs

    w = get_work(raw_id)
    if w:
        refs = w.get("referenced_works") or []
        ids = []
        for r in refs:
            if isinstance(r, str):
                rid = r.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
                if _is_doi(rid):
                    if rid not in ids:
                        ids.append(rid)
                else:
                    wid = r.rstrip("/").split("/")[-1]
                    if wid.upper().startswith("W"):
                        wid = wid[1:]
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
    """Get a single work by id (OpenAlex W123, arxiv 1908.02905, DOI, or URL).
    Results are cached per-process to avoid duplicate API calls for the same paper."""
    if work_id in _GET_WORK_CACHE:
        return _GET_WORK_CACHE[work_id]

    wid = work_id
    if wid.startswith("http"):
        wid = wid.rstrip("/").split("/")[-1]
    if wid.upper().startswith("W"):
        wid = wid[1:]

    result = None

    # DOI：优先 Crossref，减轻 OpenAlex 压力
    if _is_doi(wid) or _is_doi(work_id):
        w = crossref_get_work_by_doi(wid if _is_doi(wid) else work_id)
        if w:
            result = w

    # arxiv ID: OpenAlex 优先，429 时尝试 Crossref 查询
    if result is None and _is_arxiv_id(wid):
        r = _openalex_get(f"{OPENALEX_BASE}/works/arxiv:{wid}")
        if r and r.status_code == 200:
            result = r.json()
        elif r and r.status_code == 429:
            w = crossref_query_by_arxiv(wid)
            if w:
                result = w

    # OpenAlex W ID
    if result is None and not _is_doi(wid) and not _is_arxiv_id(wid):
        url = f"{OPENALEX_BASE}/works/W{wid}"
        r = _openalex_get(url)
        if r and r.status_code == 200:
            result = r.json()
        elif r and r.status_code == 429:
            print(f"OpenAlex get_work {work_id}: 429 rate limit (retries exhausted)")
        elif r is not None:
            print(f"OpenAlex get_work {work_id} error: {r.status_code}")

    # Evict oldest entries if cache gets too large
    if len(_GET_WORK_CACHE) > 5000:
        for k in list(_GET_WORK_CACHE)[:1000]:
            del _GET_WORK_CACHE[k]
    _GET_WORK_CACHE[work_id] = result
    return result


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
    """Convert OpenAlex/Crossref work to candidate dict (title, abstract, authors, paper_id)."""
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
    paper_id = w.get("paper_id") or ((w.get("id") or "").split("/")[-1]) or w.get("id")
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


def get_seed_paper_info(seed_id: str, skip_api_work: bool = False) -> Tuple[Optional[Dict], Optional[dict]]:
    """Get seed paper info for current_paper and profile. Returns (current_paper_dict, openalex_work_or_none).
    When skip_api_work=True, skip the OpenAlex API call if the paper is already in RAG
    (saves API quota when only default profile/query is needed)."""
    raw_id = seed_id
    if raw_id.startswith("http"):
        raw_id = raw_id.rstrip("/").split("/")[-1]
    if raw_id.upper().startswith("W"):
        raw_id = raw_id[1:]

    rag_paper = get_paper_from_rag(raw_id) if get_paper_from_rag else None
    if rag_paper:
        current = {"id": raw_id, "title": rag_paper.get("title", ""), "abstract": rag_paper.get("abstract", "")}
        work = None if skip_api_work else get_work(raw_id)
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
    """Resolve one positive candidate from ref list: RAG first, else 缓存, else API."""
    rid = ref_id
    if rid.startswith("http"):
        rid = rid.rstrip("/").split("/")[-1]
    # 查 RAG 时用原始 id（arxiv 存的是 1908.02905 格式，无 W 前缀）
    lookup_id = rid[1:] if rid.upper().startswith("W") and _is_arxiv_id(rid[1:]) else rid

    # 1) RAG
    p = get_paper_from_rag(lookup_id) if get_paper_from_rag else None
    if p:
        return {"paper_id": lookup_id, "title": p.get("title", ""), "abstract": p.get("abstract", ""), "authors": p.get("authors", [])}

    # 2) 本地缓存
    if _paper_cache:
        for key in (lookup_id, rid, _normalize_paper_id(rid)):
            if key:
                meta = _paper_cache.get_meta(key)
                if meta:
                    if add_paper_to_rag and meta:
                        add_paper_to_rag(meta)
                    return meta

    # 3) API
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


def get_negative_candidates_from_ref_pool(
    ref_pool_by_seed: Dict[str, List[str]],
    current_seed_id: str,
    current_ref_ids: List[str],
    num_total: int = NUM_NEGATIVE_HARD + NUM_NEGATIVE_EASY,
) -> List[Tuple[Dict[str, Any], float]]:
    """Get negative candidates from ref lists of OTHER seeds (refs-only mode).
    Returns [(candidate_dict, sim_score), ...]. sim_score from embedding if in RAG, else 0.0."""
    def _norm(r: str) -> str:
        x = _normalize_paper_id(r)
        return x[1:] if x and x.upper().startswith("W") and _is_arxiv_id(x[1:]) else (x or "")
    canon_seed = _norm(current_seed_id)
    exclude = {_norm(x) for x in [current_seed_id] + list(current_ref_ids) if x}
    neg_pool: List[str] = []
    seen = set()
    for sid, refs in ref_pool_by_seed.items():
        if _norm(sid) == canon_seed:
            continue
        for r in refs:
            lookup = _norm(r)
            if lookup and lookup not in exclude and lookup not in seen:
                neg_pool.append(lookup)
                seen.add(lookup)
    if len(neg_pool) < num_total:
        return []
    sampled = random.sample(neg_pool, num_total)
    seed_emb = get_embedding_for_paper(current_seed_id) if get_embedding_for_paper else None
    out = []
    for rid in sampled:
        c = resolve_positive_candidate(rid)
        if c:
            sim = 0.0
            if seed_emb:
                emb = get_embedding_for_paper(c.get("paper_id")) if get_embedding_for_paper else None
                if emb:
                    sim = max(0.0, min(1.0, cosine_similarity(seed_emb, emb)))
            out.append((c, sim))
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


def _default_profile_and_query(
    title: str,
    abstract: str,
    available_tags: List[str],
) -> Tuple[dict, str]:
    """默认 profile 和 query（无 LLM 调用）"""
    profile = {
        "position": "researcher",
        "goal": "find related work",
        "professional_level": "intermediate",
        "interest_field": available_tags[:3] if available_tags else ["machine learning", "nlp", "llm"],
    }
    query = f"Recommend papers similar to \"{title[:80]}\" or in the same field"
    return profile, query


# -----------------------------------------------------------------------------
# Main: build one listwise sample (RAG + 2 pos / 4 neg + weighted label)
# -----------------------------------------------------------------------------
def build_one_listwise_sample(
    seed_id: str,
    mode: str = "formal",
    rag_only_positives: bool = False,
    refs_only: bool = False,
    ref_pool_by_seed: Optional[Dict[str, List[str]]] = None,
) -> Optional[dict]:
    """Build one listwise sample: seed from RAG, ref list from API/cache, 2 pos + 4 neg, weighted label.
    若 rag_only_positives=True，仅用 RAG 内论文作正例，不足 2 个则跳过（不调 API 拉取元数据）。
    若 refs_only=True，负例从其他 seed 的引用池采样（不用 RAG 相似度检索）。"""
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

    user_profile, query = _default_profile_and_query(title, abstract, available_tags)
    user_profile = normalize_interest_fields(user_profile, available_tags)

    canonical_id = current_paper.get("id") or seed_id
    # 2) Reference list (cache or API)
    raw_id = _normalize_paper_id(canonical_id)
    if rag_only_positives and _paper_cache and not _paper_cache.has_refs(raw_id):
        return None  # 无缓存则跳过，避免 API 取 ref list
    ref_ids = get_reference_list_by_paper_id(canonical_id)
    if len(ref_ids) < NUM_POSITIVE:
        return None

    # 3) Two positives from ref list
    pos_candidates = []
    if rag_only_positives:
        # 仅用 RAG 内论文作正例，不足则跳过（不调 API）
        for rid in ref_ids:
            lookup_id = rid[1:] if rid.upper().startswith("W") and _is_arxiv_id(rid[1:]) else rid
            p = get_paper_from_rag(lookup_id) if get_paper_from_rag else None
            if p:
                pos_candidates.append({
                    "paper_id": lookup_id,
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "authors": p.get("authors", []),
                })
                if len(pos_candidates) >= NUM_POSITIVE:
                    break
    else:
        # RAG first, else cache/API
        for rid in ref_ids[:NUM_POSITIVE * 2]:
            c = resolve_positive_candidate(rid)
            if c:
                pos_candidates.append(c)
                if len(pos_candidates) >= NUM_POSITIVE:
                    break
    if len(pos_candidates) < NUM_POSITIVE:
        return None
    pos_candidates = pos_candidates[:NUM_POSITIVE]
    ref_set = set(ref_ids)

    # 4) Seed embedding and four negatives
    seed_emb = get_embedding_for_paper(canonical_id) if get_embedding_for_paper else None
    if not seed_emb:
        return None
    if refs_only and ref_pool_by_seed:
        neg_with_sim = get_negative_candidates_from_ref_pool(
            ref_pool_by_seed, canonical_id, ref_ids, num_total=NUM_NEGATIVE_HARD + NUM_NEGATIVE_EASY
        )
    else:
        neg_with_sim = get_negative_candidates_from_rag(
            seed_emb, canonical_id, ref_ids, num_hard=NUM_NEGATIVE_HARD, num_easy=NUM_NEGATIVE_EASY
        )
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

    # 6) Label = 0.7 * cited + 0.3 * sim（无 LLM）
    labels = []
    for i in range(len(candidates)):
        lab = WEIGHT_CITED * cited_flags[i] + WEIGHT_SIM * sim_scores[i]
        labels.append(round(lab, 4))

    return {
        "current_paper": {"id": current_paper.get("id"), "title": current_paper.get("title"), "abstract": current_paper.get("abstract")},
        "user_profile": user_profile,
        "query": query,
        "candidate_list": candidates,
        "label": labels,
    }


def _worker_build_samples(
    args_tuple: Tuple[List[str], str, int, Optional[str], bool, bool, Optional[Dict[str, List[str]]], int],
) -> List[dict]:
    """Worker: 对一批 seed_ids 生成 samples。返回 sample 列表。"""
    seed_ids, mode, worker_id, cache_dir, rag_only, refs_only, ref_pool_by_seed, n_workers = args_tuple
    global _paper_cache, _OPENALEX_MIN_GAP
    # Scale rate limit gap by number of workers to stay under OpenAlex ~10 req/s
    _OPENALEX_MIN_GAP = max(0.2, 0.15 * n_workers)
    if cache_dir:
        try:
            from paper_cache import PaperCache
            _paper_cache = PaperCache(cache_dir)
        except Exception:
            _paper_cache = None
    samples = []
    for i, sid in enumerate(seed_ids):
        one = build_one_listwise_sample(
            sid,
            mode=mode,
            rag_only_positives=rag_only,
            refs_only=refs_only,
            ref_pool_by_seed=ref_pool_by_seed,
        )
        if one:
            samples.append(one)
        if (i + 1) % 10 == 0 or one:
            print(f"[worker-{worker_id}] {i+1}/{len(seed_ids)} seeds, {len(samples)} samples", flush=True)
    return samples


def main():
    global _paper_cache
    import argparse
    parser = argparse.ArgumentParser(
        description="Listwise 多进程并行生成，label=0.7*cited+0.3*sim，无LLM"
    )
    parser.add_argument("--mode", type=str, choices=["test", "formal"], default="formal")
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("-o", "--out", type=str, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=4, help="并行进程数 (default: 4)")
    parser.add_argument("--cache-dir", type=str, default="",
                        help="本地缓存目录（由 prefetch_paper_cache 生成），使用后几乎无 API 调用")
    parser.add_argument("--rag-only-positives", action="store_true",
                        help="仅用 RAG 内论文作正例，不足 2 个则跳过 seed（不调 API 拉取引用论文元数据）")
    parser.add_argument("--refs-only", action="store_true",
                        help="负例仅从其他 seed 的引用池采样（不用 RAG 相似度检索）")
    parser.add_argument("--offset", type=int, default=0,
                        help="Chroma ID offset — 跳过前 N 篇，从不同区域取种子")
    parser.add_argument("--pool-size", type=int, default=0,
                        help="从 Chroma 取 pool-size 篇再随机采样 seed（0 = 不启用，直接取 limit 篇）")
    parser.add_argument("--exclude-file", type=str, nargs="*", default=[],
                        help="排除这些 JSON 数据文件中已用过的 seed（避免重复生成）")
    parser.add_argument("--openalex-email", type=str, default="")
    args = parser.parse_args()

    if args.num is None:
        args.num = TEST_NUM_SAMPLES if args.mode == "test" else NUM_SAMPLES
    output_path = args.out if args.out else DEFAULT_OUTPUT_PATH
    n_jobs = max(1, args.jobs)

    global OPENALEX_EMAIL
    if args.openalex_email:
        OPENALEX_EMAIL = args.openalex_email

    _paper_cache = None
    if args.cache_dir:
        try:
            from paper_cache import PaperCache
            _paper_cache = PaperCache(args.cache_dir)
            print(f"[cache] 使用本地缓存: {args.cache_dir} (refs={_paper_cache.refs_count()}, meta={_paper_cache.meta_count()})")
        except Exception as e:
            print(f"[cache] 加载失败 {e}，将使用 API")

    rag_only = getattr(args, "rag_only_positives", False)
    refs_only = getattr(args, "refs_only", False)
    if rag_only:
        print("[rag-only] 仅用 RAG 内论文作正例，不足则跳过 seed")
    if refs_only:
        print("[refs-only] 负例从其他 seed 的引用池采样，不用 RAG 相似度检索")
    # Load seeds to exclude from previous data files
    exclude_seeds = set()
    for ef in (args.exclude_file or []):
        try:
            with open(ef, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            for s in d.get("samples", []):
                sid = (s.get("current_paper") or {}).get("id", "")
                if sid:
                    exclude_seeds.add(sid)
        except Exception as e:
            print(f"[exclude] Failed to load {ef}: {e}")
    if exclude_seeds:
        print(f"[exclude] Will skip {len(exclude_seeds)} seeds from {len(args.exclude_file)} file(s)")

    pool_size = args.pool_size if args.pool_size > 0 else max(args.num * 5, 10000)
    print(f"[mode={args.mode}] Generating {args.num} samples, {n_jobs} workers, label=0.7*cited+0.3*sim")
    print(f"[seeds] offset={args.offset}, pool_size={pool_size}")
    seed_ids = []
    if get_seed_paper_ids_from_rag:
        seed_ids = get_seed_paper_ids_from_rag(
            limit=max(args.num + 200, 1500),
            offset=args.offset,
            pool_size=pool_size,
            shuffle=True,
        )
    if not seed_ids:
        print("No seed paper IDs from RAG. Ensure vector DB is populated.")
        return

    if exclude_seeds:
        before = len(seed_ids)
        seed_ids = [s for s in seed_ids if s not in exclude_seeds]
        print(f"[exclude] Filtered {before} -> {len(seed_ids)} seeds after dedup")

    ref_pool_by_seed = None
    if refs_only:
        print("[refs-only] Pre-fetching ref lists for all seeds...")
        ref_pool_by_seed = {}
        for i, sid in enumerate(seed_ids):
            refs = get_reference_list_by_paper_id(sid)
            if len(refs) >= NUM_POSITIVE:
                ref_pool_by_seed[sid] = refs
            if (i + 1) % 200 == 0:
                print(f"  Fetched {i + 1}/{len(seed_ids)} ref lists")
        print(f"[refs-only] Got ref lists for {len(ref_pool_by_seed)} seeds")

    # 分片给各 worker（传入 cache_dir、rag_only、refs_only、ref_pool_by_seed）
    chunk_size = max(1, (len(seed_ids) + n_jobs - 1) // n_jobs)
    cache_dir = args.cache_dir if args.cache_dir else None
    chunks = [
        (seed_ids[i : i + chunk_size], args.mode, wid, cache_dir, rag_only, refs_only, ref_pool_by_seed, n_jobs)
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
