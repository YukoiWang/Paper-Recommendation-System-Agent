"""Intent cascade: rules → embedding kNN → LLM fallback (design §3.2)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INTENTS = [
    "chitchat", "meta", "feedback", "daily", "recommend",
    "explain", "compare", "survey", "factoid", "followup",
]

_GREET = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|你好|谢谢|好的|再见)[\s!！。.?？]*$",
    re.I,
)
_META = re.compile(r"(你是谁|你能做什么|what can you do|who are you|what are you)", re.I)
_FEEDBACK = re.compile(r"(太老|不要综述|不喜欢|换一批|too old|don't like|not these)", re.I)
_FOLLOW = re.compile(
    r"(第二篇|第一篇|第三篇|展开讲|那篇|那一篇|more about|the first paper|the second paper)",
    re.I,
)
_COMPARE = re.compile(r"(\bvs\b|versus|对比|比较|区别|有什么不同|difference between)", re.I)
_RECOMMEND = re.compile(r"(推荐几篇|找几篇|给我推荐|recommend papers|find me (some )?papers|suggest papers)", re.I)
_EXPLAIN = re.compile(r"(讲讲|介绍一下|怎么演进|what is\b|how does\b|explain\b)", re.I)
_SURVEY = re.compile(r"(综述|文献综述|\bsurvey\b|全面回顾)", re.I)
_FACTOID = re.compile(
    r"(谁一作|哪年|哪一年|什么数据集|这篇的方法|first author|what dataset|which year|what method did (this|the) paper)",
    re.I,
)
_MORE_LIKE = re.compile(r"(再找几篇类似|再推荐类似|similar papers|more like these)", re.I)
_LATEST = re.compile(r"(最新|最近|sota|state of the art|latest|recent|newest)", re.I)


def classify_intent(
    query: str,
    *,
    has_cited: bool = False,
    is_daily: bool = False,
    embedder=None,
    llm_call=None,
    tau: float = 0.42,
    delta: float = 0.04,
) -> Tuple[str, str, float]:
    """Return (intent, source, confidence)."""
    q = (query or "").strip()
    if is_daily or not q:
        return "daily", "rule", 1.0

    hit = _rule_intent(q, has_cited=has_cited)
    if hit:
        return hit[0], "rule", 1.0

    emb_hit = _embedding_intent(q, embedder, tau=tau, delta=delta)
    if emb_hit:
        return emb_hit[0], "embedding", float(emb_hit[1])

    if llm_call:
        try:
            intent = _llm_intent(q, has_cited, llm_call)
            if intent in INTENTS:
                return intent, "llm", 0.7
        except Exception as e:
            logger.warning("intent LLM fallback failed: %s", e)
    return "explain", "llm", 0.4


def _rule_intent(q: str, has_cited: bool) -> Optional[Tuple[str, str]]:
    if _GREET.match(q) or len(q) <= 1:
        return "chitchat", "rule"
    if _META.search(q):
        return "meta", "rule"
    if _FEEDBACK.search(q):
        return "feedback", "rule"
    if has_cited and _MORE_LIKE.search(q):
        return None
    if has_cited and _FOLLOW.search(q):
        return "followup", "rule"
    if _COMPARE.search(q):
        return "compare", "rule"
    if _SURVEY.search(q):
        return "survey", "rule"
    if _FACTOID.search(q):
        return "factoid", "rule"
    if _RECOMMEND.search(q):
        return "recommend", "rule"
    if _EXPLAIN.search(q) and not _COMPARE.search(q):
        return "explain", "rule"
    return None


_PROTO_CACHE: Optional[Tuple[List[str], np.ndarray, List[str]]] = None


def _load_prototypes() -> Tuple[List[str], List[str]]:
    path = Path(__file__).parent / "prototypes.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    return [x["intent"] for x in items], [x["text"] for x in items]


def _embedding_intent(q: str, embedder, tau: float, delta: float) -> Optional[Tuple[str, float]]:
    if embedder is None:
        return None
    global _PROTO_CACHE
    try:
        qv = np.asarray(embedder.encode(q), dtype=np.float32).ravel()
        n = np.linalg.norm(qv)
        if n > 0:
            qv = qv / n
        if _PROTO_CACHE is None:
            labels, texts = _load_prototypes()
            if hasattr(embedder, "encode_batch"):
                mat = np.asarray(embedder.encode_batch(texts), dtype=np.float32)
            else:
                mat = np.stack([np.asarray(embedder.encode(t), dtype=np.float32) for t in texts])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-10)
            _PROTO_CACHE = (labels, mat, texts)
        labels, mat, _ = _PROTO_CACHE
        scores = mat @ qv
        per: Dict[str, List[float]] = {}
        for lab, s in zip(labels, scores):
            per.setdefault(lab, []).append(float(s))
        ranked = sorted(
            ((lab, float(np.mean(sorted(v, reverse=True)[:3]))) for lab, v in per.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            return None
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        if s1 >= tau and (s1 - s2) >= delta:
            return best, s1
    except Exception as e:
        logger.warning("intent embedding match failed: %s", e)
    return None


def _llm_intent(q: str, has_cited: bool, llm_call) -> str:
    prompt = (
        "Classify the academic-assistant user query into ONE label:\n"
        + ", ".join(INTENTS)
        + f"\nHas cited papers in session: {has_cited}\n"
        "Query: " + q[:500] + "\n"
        'Return JSON only: {"intent": "<label>"}'
    )
    raw = llm_call(prompt)
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    return str(data.get("intent", "explain")).lower().strip()


def detect_need_sota(query: str) -> bool:
    return bool(_LATEST.search(query or ""))


def extract_compare_entities(query: str) -> List[str]:
    q = query or ""
    parts = re.split(r"\s+vs\.?\s+|versus|\band\b|对比|比较|和|与|以及", q, flags=re.I)
    ents = []
    noise = {"的区别", "有什么不同", "区别", "论文", "papers", "the", "a"}
    for p in parts:
        p = re.sub(r"(讲讲|介绍|对比|比较|推荐|区别|differences?)", "", p, flags=re.I)
        p = p.strip(" ？?。,.，")
        if 1 < len(p) <= 40 and p.lower() not in noise:
            ents.append(p)
    # keep unique order
    seen = set()
    out = []
    for e in ents:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out[:4]


def topic_from_query(query: str) -> str:
    t = (query or "").strip()
    t = re.sub(r"^(推荐几篇|找几篇|讲讲|介绍一下|帮我)\s*", "", t)
    return t[:200] if t else (query or "")


def topic_is_grounded(topic: str, query: str, synonyms: Optional[Dict[str, str]] = None) -> bool:
    """Reject topics that add named tokens not in query (design §4.2)."""
    if not topic:
        return True
    synonyms = synonyms or {"rag": "检索增强", "retrieval-augmented": "检索增强"}
    q = (query or "").lower()
    import re as _re
    tokens = _re.findall(r"[a-zA-Z][\w-]*|[\u4e00-\u9fff]{2,}", topic)
    for tok in tokens:
        low = tok.lower()
        if low in {"the", "and", "for", "with", "paper", "papers", "相关", "论文"}:
            continue
        if tok in query or low in q:
            continue
        if any(low in v.lower() or low in k for k, v in synonyms.items()):
            continue
        if len(tok) >= 2 and tok.lower() not in q:
            # allow English expansion of Chinese query
            if any("\u4e00" <= c <= "\u9fff" for c in query):
                continue
            return False
    return True
