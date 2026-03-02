#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic RAG evaluation dataset generation.

目标：生成用于评估 RAG 的合成数据集：
  { query + ground_truth_contexts + context_summary(系统生成) + answer(系统生成) }

约束：
- 用于生成 query 的种子论文必须在向量库 / 元数据 DB 里（RAG 已收录）
- ground_truth_contexts 中的所有论文也都在向量库 / 元数据 DB 里

支持三类 query：
1) 一般推荐型（general_recommendation）
   - 子类型：模糊（fuzzy）、错误（noisy）、正常推荐（normal）、复杂需求（complex）
2) 具体方法询问型（specific_method）
3) 综述总结型（survey_overview）

依赖：
- 使用 backend.services.* 提供的 RAG 服务（经 rag_listwise_helpers 间接调用）
- 使用 Qwen / DeepSeek（OpenAI 兼容接口）调用大模型生成 query / context / answer

用法示例：
  python data/create_rag_eval_dataset.py --mode mixed -n 50 -o ~/rag_eval/rag_eval.json
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# 项目路径与现有辅助工具
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# 优先使用 pysqlite3 以满足 ChromaDB 对 sqlite 版本要求
try:
    import pysqlite3  # type: ignore

    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

from agent.models import Paper  # noqa: E402
from rag_listwise_helpers import (  # noqa: E402
    add_paper_to_rag,
    cosine_similarity,
    get_embedding_for_paper,
    get_paper_from_rag,
    get_seed_paper_ids_from_rag,
    similarity_search_rag,
)

# 复用 create_listwise2 中已经实现好的 OpenAlex / CrossRef 工具（如可用）
try:
    from create_listwise2 import (  # type: ignore  # noqa: E402
        get_reference_list_by_paper_id,
        get_work,
    )
except Exception:
    get_reference_list_by_paper_id = None  # type: ignore
    get_work = None  # type: ignore


# ---------------------------------------------------------------------------
# LLM 配置与工具
# ---------------------------------------------------------------------------
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
QWEN_BASE = (os.getenv("QWEN_API_BASE") or os.getenv("DASHSCOPE_API_BASE") or "").rstrip("/")
if not QWEN_BASE:
    QWEN_BASE = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = os.getenv("QWEN_MODEL") or "qwen-plus"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = (os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"


def _chat_openai_compatible(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """兼容 OpenAI / DashScope / DeepSeek 的简单 chat helper。"""
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
                "max_tokens": max_tokens,
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
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中尽力提取 JSON（去掉 ```json 包裹等）。"""
    text = (text or "").strip()
    for pattern in [r"```(?:json)?\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"]:
        m = re.search(pattern, text)
        if m:
            text = m.group(1).strip()
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _call_qwen(messages: List[dict], temperature: float = 0.5, max_tokens: int = 1024) -> Optional[str]:
    if not QWEN_API_KEY:
        return None
    try:
        return _chat_openai_compatible(QWEN_API_KEY, QWEN_BASE, QWEN_MODEL, messages, temperature, max_tokens)
    except Exception:
        return None


def _call_deepseek(messages: List[dict], temperature: float = 0.3, max_tokens: int = 1024) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        return _chat_openai_compatible(DEEPSEEK_API_KEY, DEEPSEEK_BASE, DEEPSEEK_MODEL, messages, temperature, max_tokens)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class EvalSample:
    query_type: str  # general_recommendation | specific_method | survey_overview
    query_subtype: Optional[str]  # fuzzy | noisy | normal | complex | None
    query: str
    seed_paper_id: str
    seed_paper_title: str
    seed_paper_abstract: str
    ground_truth_paper_ids: List[str]
    ground_truth_contexts: List[Dict[str, Any]]
    context_summary: str
    answer: str


# ---------------------------------------------------------------------------
# 基础工具：从 RAG 中取论文、向量检索
# ---------------------------------------------------------------------------
def _paper_from_meta(p: Dict[str, Any]) -> Paper:
    return Paper(
        paper_id=p.get("paper_id") or p.get("id") or "",
        title=p.get("title", ""),
        abstract=p.get("abstract", ""),
        authors=p.get("authors", []) or [],
    )


def _get_chroma_collection():
    """Lazy-init direct Chroma collection for metadata fallback."""
    # 避免循环导入，按需导入 chromadb
    try:
        import chromadb  # type: ignore
        from chromadb.config import Settings as ChromaSettings  # type: ignore
    except Exception:
        return None
    path = os.getenv("CHROMA_PERSIST_DIR") or os.path.expanduser("~/chroma_db")
    try:
        client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
        return client.get_collection("papers")
    except Exception as e:
        print(f"Chroma collection load error: {e}")
        return None


def _get_paper_from_chroma(paper_id: str) -> Optional[Dict[str, Any]]:
    """直接从 Chroma collection 里取一篇论文的元数据，用于 metadata_db 缺失时的兜底。"""
    coll = _get_chroma_collection()
    if coll is None:
        return None
    try:
        res = coll.get(ids=[paper_id], include=["metadatas", "documents"])
    except Exception:
        return None
    ids = res.get("ids") or []
    if not ids:
        return None
    metas = (res.get("metadatas") or [{}])
    docs = (res.get("documents") or [""])
    meta = metas[0] or {}
    doc = docs[0] or ""
    return {
        "paper_id": paper_id,
        "title": meta.get("title", ""),
        "abstract": doc,
        "authors": meta.get("authors", []) or [],
        "venue": meta.get("venue", ""),
        "year": meta.get("year"),
    }


def _get_seed_papers(limit: int = 1000) -> List[Paper]:
    ids = get_seed_paper_ids_from_rag(limit=limit) or []
    out: List[Paper] = []
    for pid in ids:
        meta = get_paper_from_rag(pid) or _get_paper_from_chroma(pid)
        if not meta:
            continue
        p = _paper_from_meta({"paper_id": pid, **meta})
        if p.title and p.abstract:
            out.append(p)
    return out


def _get_papers_by_ids(ids: List[str]) -> List[Paper]:
    out: List[Paper] = []
    for pid in ids:
        meta = get_paper_from_rag(pid) or _get_paper_from_chroma(pid)
        if not meta:
            continue
        out.append(_paper_from_meta({"paper_id": pid, **meta}))
    return out


def _semantic_neighbors(paper_id: str, top_k: int = 50, exclude: Optional[List[str]] = None) -> List[Paper]:
    emb = get_embedding_for_paper(paper_id)
    if not emb:
        return []
    raw = similarity_search_rag(emb, top_k=top_k, exclude_ids=list(exclude or []) + [paper_id]) or []
    out: List[Paper] = []
    for r in raw:
        pid = r.get("paper_id")
        if not pid:
            continue
        meta = get_paper_from_rag(pid) or _get_paper_from_chroma(pid)
        if not meta:
            continue
        out.append(_paper_from_meta({"paper_id": pid, **meta}))
    return out


# ---------------------------------------------------------------------------
# 一般推荐型：query 生成 & 候选池
# ---------------------------------------------------------------------------
def _generate_general_queries(paper: Paper) -> List[Dict[str, str]]:
    """
    对于一篇论文，生成 4 种类型的一般推荐型 query：
    - fuzzy: 模糊、需要扩展
    - noisy: 含有一定误解 / 错误
    - normal: 正常的推荐需求
    - complex: 复杂需求、多条件 / 需要分解
    """
    sys_msg = {
        "role": "system",
        "content": (
            "你是一个科研论文推荐系统的产品经理，擅长把论文元数据转成真实用户会问的问题。\n"
            "根据给定的论文标题和摘要，生成 4 条中文查询：\n"
            "1) 模糊：只大致描述兴趣方向，需要系统做语义扩展；\n"
            "2) 错误：带有一定误解或细节错误，但仍能大致看出用户想找什么；\n"
            "3) 正常：清晰表达“想找类似论文”的一般推荐需求；\n"
            "4) 复杂：包含多条约束或子需求，需要系统分解问题。\n"
            "输出 JSON，格式为：\n"
            "{\n"
            '  "queries": [\n'
            '    {"subtype": "fuzzy",   "query": "..."},\n'
            '    {"subtype": "noisy",   "query": "..."},\n'
            '    {"subtype": "normal",  "query": "..."},\n'
            '    {"subtype": "complex", "query": "..."}\n'
            "  ]\n"
            "}\n"
            "只返回 JSON，不要额外解释。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"论文标题：{paper.title}\n\n"
            f"论文摘要：{paper.abstract}\n\n"
            "请根据这篇论文生成上述 4 条查询。"
        ),
    }
    text = _call_qwen([sys_msg, user_msg], temperature=0.8, max_tokens=512)
    if not text:
        return []
    obj = _extract_json(text)
    if not obj:
        return []
    items = obj.get("queries") or []
    out: List[Dict[str, str]] = []
    for it in items:
        q = (it.get("query") or "").strip()
        st = (it.get("subtype") or "").strip()
        if q and st:
            out.append({"subtype": st, "query": q})
    return out


def _build_candidate_pool_general(seed: Paper, max_refs: int = 20, max_semantic: int = 10) -> List[Paper]:
    """
    构建高质量候选池：
    a) 引用网络召回：OpenAlex 引用列表 + 通过 RAG filter & 向量相似度排序，取前 max_refs
    b) 语义扩展召回：RAG ANN 搜索，取前 max_semantic
    """
    candidates: Dict[str, Paper] = {}

    # (a) 引用网络
    if get_reference_list_by_paper_id:
        try:
            ref_ids = get_reference_list_by_paper_id(seed.paper_id)
        except Exception:
            ref_ids = []
        ref_ids = ref_ids or []
        random.shuffle(ref_ids)

        seed_emb = get_embedding_for_paper(seed.paper_id)
        scored: List[Tuple[float, Paper]] = []
        if seed_emb:
            for rid in ref_ids:
                # 引用 ID 大多不是 RAG 内部 ID，通过 add_paper_to_rag 补齐向量库
                meta = get_paper_from_rag(rid) or _get_paper_from_chroma(rid)
                if not meta:
                    # 尝试走 OpenAlex / S2 -> add_paper_to_rag
                    if get_work:
                        try:
                            w = get_work(rid)
                        except Exception:
                            w = None
                        if w:
                            try:
                                title = (w.get("title") or w.get("display_name") or "").strip()
                                abstract = ""
                                inv = w.get("abstract_inverted_index")
                                if isinstance(inv, dict) and inv:
                                    pairs = []
                                    for word, pos_list in inv.items():
                                        for pos in pos_list:
                                            pairs.append((pos, word))
                                    pairs.sort(key=lambda x: x[0])
                                    abstract = " ".join(p[1] for p in pairs)[:800]
                                paper_id = (w.get("id") or "").split("/")[-1]
                                authors = []
                                for a in w.get("authorships") or []:
                                    au = a.get("author") or {}
                                    name = au.get("display_name") or a.get("raw_author_name")
                                    if name:
                                        authors.append(name)
                                payload = {
                                    "paper_id": paper_id,
                                    "title": title,
                                    "abstract": abstract,
                                    "authors": authors[:10],
                                }
                                if add_paper_to_rag(payload):
                                    meta = get_paper_from_rag(paper_id)
                            except Exception:
                                meta = None
                if not meta:
                    continue
                p = _paper_from_meta({"paper_id": meta.get("paper_id", rid), **meta})
                if not p.abstract or not p.title:
                    continue
                if p.paper_id == seed.paper_id:
                    continue
                emb = get_embedding_for_paper(p.paper_id)
                if not emb:
                    continue
                sim = cosine_similarity(seed_emb, emb)
                scored.append((sim, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, p in scored[:max_refs]:
            candidates[p.paper_id] = p

    # (b) 语义扩展
    neighbors = _semantic_neighbors(seed.paper_id, top_k=max_semantic * 3, exclude=[seed.paper_id] + list(candidates))
    for p in neighbors:
        if p.paper_id in candidates:
            continue
        candidates[p.paper_id] = p
        if len(candidates) >= max_refs + max_semantic:
            break

    return list(candidates.values())


def _llm_filter_candidates_for_query(
    query: str,
    seed: Paper,
    candidates: List[Paper],
    min_keep: int = 10,
    max_keep: int = 15,
) -> List[Paper]:
    """调用 LLM 对候选论文进行二次筛选，返回最契合 query 的若干篇。"""
    if not candidates:
        return []

    items = []
    for idx, p in enumerate(candidates):
        items.append(
            {
                "idx": idx,
                "title": p.title,
                "abstract": p.abstract[:700],
            }
        )

    sys_msg = {
        "role": "system",
        "content": (
            "你是一个学术论文推荐系统，任务是根据用户查询和一篇种子论文，从候选论文中挑选最相关的 10–15 篇作为“理想推荐列表”。\n"
            "请综合考虑：主题相关性、方法相近性、应用场景、时间新旧等。\n"
            "只输出 JSON，格式如下：\n"
            "{ \"selected_indices\": [0, 3, 5, ...] }\n"
            "其中 selected_indices 是整数下标列表，长度在 10 到 15 之间；如果候选不足，就尽量多选。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"用户查询：{query}\n\n"
            f"种子论文标题：{seed.title}\n\n"
            f"种子论文摘要：{seed.abstract}\n\n"
            f"候选论文列表（JSON）：\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        ),
    }

    text = _call_deepseek([sys_msg, user_msg], temperature=0.2, max_tokens=512) or _call_qwen(
        [sys_msg, user_msg], temperature=0.3, max_tokens=512
    )
    if not text:
        # fallback：不走 LLM，直接取前若干篇
        return candidates[:max_keep]
    obj = _extract_json(text)
    if not obj:
        return candidates[:max_keep]
    raw = obj.get("selected_indices") or []
    try:
        idxs = [int(i) for i in raw]
    except Exception:
        return candidates[:max_keep]
    idxs = [i for i in idxs if 0 <= i < len(candidates)]
    if not idxs:
        return candidates[:max_keep]
    if len(idxs) < min_keep:
        extra = [i for i in range(len(candidates)) if i not in idxs]
        idxs = idxs + extra[: max_keep - len(idxs)]
    return [candidates[i] for i in idxs[:max_keep]]


def _generate_answer_from_context(query: str, papers: List[Paper], mode: str) -> Tuple[str, str]:
    """
    基于 ground_truth_contexts 生成：
    - context_summary：结构化、可作为 RAG context 的压缩摘要
    - answer：面向用户的最终回答
    """
    if not papers:
        return "", ""
    brief_list = []
    for i, p in enumerate(papers, start=1):
        brief_list.append(f"{i}. 标题：{p.title}\n   摘要：{p.abstract[:800]}")
    papers_block = "\n\n".join(brief_list)

    sys_msg = {
        "role": "system",
        "content": (
            "你是一个科研助手，需要根据给定的多篇论文，为用户问题生成高质量的检索增强回答。\n"
            "首先用较为客观、中立的学术语言，总结这些论文中与问题最相关的关键信息，形成一段“内部用的上下文总结”；\n"
            "然后基于该上下文，用更口语化、对学生友好的方式回答用户问题。\n"
            "输出 JSON，格式为：\n"
            "{\n"
            '  \"context_summary\": \"...\",\n'
            '  \"answer\": \"...\"\n'
            "}\n"
            "只返回 JSON，不要额外文本。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"用户问题：{query}\n\n"
            f"相关论文（仅供参考）：\n{papers_block}\n\n"
            "请按上述要求生成 context_summary 和 answer。"
        ),
    }

    text = _call_qwen([sys_msg, user_msg], temperature=0.4, max_tokens=1024) or _call_deepseek(
        [sys_msg, user_msg], temperature=0.4, max_tokens=1024
    )
    if not text:
        return "", ""
    obj = _extract_json(text) or {}
    ctx = (obj.get("context_summary") or "").strip()
    ans = (obj.get("answer") or "").strip()
    return ctx, ans


def build_general_samples_from_seed(seed: Paper) -> List[EvalSample]:
    """给定一篇种子论文，构造若干一般推荐型样本（4 个子类型）。"""
    queries = _generate_general_queries(seed)
    if not queries:
        return []
    pool = _build_candidate_pool_general(seed, max_refs=20, max_semantic=10)
    if not pool:
        return []
    samples: List[EvalSample] = []
    for q in queries:
        subtype = q["subtype"]
        query = q["query"]
        gt_papers = _llm_filter_candidates_for_query(query, seed, pool, min_keep=10, max_keep=15)
        if not gt_papers:
            continue
        ctx, ans = _generate_answer_from_context(query, gt_papers, mode="general")
        if not ctx or not ans:
            continue
        samples.append(
            EvalSample(
                query_type="general_recommendation",
                query_subtype=subtype,
                query=query,
                seed_paper_id=seed.paper_id,
                seed_paper_title=seed.title,
                seed_paper_abstract=seed.abstract,
                ground_truth_paper_ids=[p.paper_id for p in gt_papers],
                ground_truth_contexts=[asdict(p) for p in gt_papers],
                context_summary=ctx,
                answer=ans,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# 具体方法询问型：gt context 就是种子论文本身
# ---------------------------------------------------------------------------
def build_specific_method_sample(seed: Paper) -> Optional[EvalSample]:
    sys_msg = {
        "role": "system",
        "content": (
            "你是一个科研问答助手，需要根据一篇具体论文生成一个“针对论文内容细节”的自然语言问题，"
            "问题要围绕该论文提出的方法、实验设置、改进点等，可以稍微细一点，但要保证从摘要和常见论文结构中可以回答。\n"
            "输出 JSON：{ \"query\": \"...\" }，只返回 JSON。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"论文标题：{seed.title}\n\n论文摘要：{seed.abstract}\n\n请生成一个针对该论文方法/实验的具体提问。",
    }
    text = _call_qwen([sys_msg, user_msg], temperature=0.6, max_tokens=256)
    if not text:
        return None
    obj = _extract_json(text) or {}
    query = (obj.get("query") or "").strip()
    if not query:
        return None

    ctx, ans = _generate_answer_from_context(query, [seed], mode="specific")
    if not ctx or not ans:
        return None
    return EvalSample(
        query_type="specific_method",
        query_subtype=None,
        query=query,
        seed_paper_id=seed.paper_id,
        seed_paper_title=seed.title,
        seed_paper_abstract=seed.abstract,
        ground_truth_paper_ids=[seed.paper_id],
        ground_truth_contexts=[asdict(seed)],
        context_summary=ctx,
        answer=ans,
    )


# ---------------------------------------------------------------------------
# 综述总结型：基于“疑似综述论文”构建上下文
# ---------------------------------------------------------------------------
TOP_VENUE_SCORES: Dict[str, float] = {
    "neurips": 1.0,
    "nips": 1.0,
    "icml": 1.0,
    "iclr": 1.0,
    "cvpr": 1.0,
    "iccv": 0.9,
    "eccv": 0.9,
    "acl": 1.0,
    "emnlp": 0.9,
    "naacl": 0.8,
    "kdd": 0.8,
    "sigir": 0.8,
}


def _is_survey_like(title: str) -> bool:
    t = (title or "").lower()
    return any(
        k in t
        for k in [
            "survey",
            "review",
            "overview",
            "a review",
            "综述",
            "调查",
        ]
    )


def _compute_importance_scores(papers: List[Paper]) -> Dict[str, float]:
    """
    使用 OpenAlex（若可用）估计每篇论文的重要性：
      Score = 0.7 * Norm(Cites_Per_Year) + 0.3 * Norm(Venue_Score)
    """
    if not get_work or not papers:
        # 退化为均匀打分
        return {p.paper_id: 0.5 for p in papers}

    stats: List[Tuple[str, float, float]] = []
    this_year = time.gmtime().tm_year

    for p in papers:
        w = None
        try:
            w = get_work(p.paper_id)
        except Exception:
            w = None
        if not w:
            stats.append((p.paper_id, 0.0, 0.0))
            continue

        cited = float(w.get("cited_by_count") or 0.0)
        year = (
            w.get("publication_year")
            or (w.get("from_publication_date") or "")[:4]
            or (w.get("publication_date") or "")[:4]
        )
        try:
            year = int(year)
        except Exception:
            year = this_year
        age = max(1, this_year - year + 1)
        cpy = cited / float(age)

        hv = w.get("host_venue") or {}
        venue_name = (hv.get("display_name") or hv.get("publisher") or "").lower()
        venue_score = 0.0
        for k, v in TOP_VENUE_SCORES.items():
            if k in venue_name:
                venue_score = max(venue_score, v)
        stats.append((p.paper_id, cpy, venue_score))

    if not stats:
        return {}

    c_list = [s[1] for s in stats]
    v_list = [s[2] for s in stats]
    c_min, c_max = min(c_list), max(c_list)
    v_min, v_max = min(v_list), max(v_list)

    def _norm(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.5
        return (x - lo) / (hi - lo + 1e-8)

    out: Dict[str, float] = {}
    for pid, cpy, vs in stats:
        score = 0.7 * _norm(cpy, c_min, c_max) + 0.3 * _norm(vs, v_min, v_max)
        out[pid] = float(score)
    return out


def _pick_survey_seed(seeds: List[Paper]) -> Optional[Paper]:
    """从种子论文中挑选一篇“看起来像综述”的论文。"""
    survey_like = [p for p in seeds if _is_survey_like(p.title)]
    if survey_like:
        return random.choice(survey_like)
    return random.choice(seeds) if seeds else None


def build_survey_sample(seed: Paper) -> Optional[EvalSample]:
    """
    综述总结型：
      - Seed_Survey: 看起来像综述的论文
      - 候选池：语义邻居 Top 50
      - 核心综述层：候选中 title 含 survey/review/... 的论文
      - 里程碑层：其余论文里按重要性分数选 Top 5–8
    """
    # Step 2: 生成用户 query
    sys_msg = {
        "role": "system",
        "content": (
            "你是研究生，需要根据一篇综述论文，向智能助手提出一个自然、口语化的问题，"
            "目的是了解某个研究方向的发展脉络和关键论文。\n"
            "输出 JSON：{ \"query\": \"...\" }，只返回 JSON。"
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"综述论文标题：{seed.title}\n\n"
            f"摘要：{seed.abstract}\n\n"
            "请生成一个类似“帮我梳理一下 XXX 领域的发展脉络和关键论文”的中文问题。"
        ),
    }
    text = _call_qwen([sys_msg, user_msg], temperature=0.6, max_tokens=256)
    if not text:
        return None
    obj = _extract_json(text) or {}
    query = (obj.get("query") or "").strip()
    if not query:
        return None

    # Step 3: 语义扩展召回候选池
    neighbors = _semantic_neighbors(seed.paper_id, top_k=80, exclude=[seed.paper_id])
    if not neighbors:
        return None
    # 只取 Top 50
    neighbors = neighbors[:50]

    # Step 4a: 核心综述层
    core_surveys = [p for p in neighbors if _is_survey_like(p.title)]

    # Step 4b: 里程碑层
    non_surveys = [p for p in neighbors if p not in core_surveys]
    imp_scores = _compute_importance_scores(non_surveys)
    non_surveys_sorted = sorted(non_surveys, key=lambda p: imp_scores.get(p.paper_id, 0.0), reverse=True)
    milestones = non_surveys_sorted[:8]

    # 合并去重
    final_ids: Dict[str, Paper] = {}
    for p in core_surveys + milestones:
        final_ids[p.paper_id] = p
    gt_papers = list(final_ids.values())
    if not gt_papers:
        return None

    ctx, ans = _generate_answer_from_context(query, gt_papers, mode="survey")
    if not ctx or not ans:
        return None

    return EvalSample(
        query_type="survey_overview",
        query_subtype=None,
        query=query,
        seed_paper_id=seed.paper_id,
        seed_paper_title=seed.title,
        seed_paper_abstract=seed.abstract,
        ground_truth_paper_ids=[p.paper_id for p in gt_papers],
        ground_truth_contexts=[asdict(p) for p in gt_papers],
        context_summary=ctx,
        answer=ans,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 评估用合成数据集生成脚本")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["general", "specific", "survey", "mixed"],
        default="mixed",
        help="要生成的 query 类型：general(一般推荐)、specific(具体方法)、survey(综述)、mixed(混合)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=20,
        help="期望生成的样本数量（mixed 模式下是总量）",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.path.expanduser("~/rag_eval/rag_eval_dataset.json"),
        help="输出 JSON 路径",
    )
    parser.add_argument(
        "--seed-limit",
        type=int,
        default=2000,
        help="从向量库中最多取多少篇论文作为种子候选",
    )
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] mode={args.mode}, num={args.num}, seed_limit={args.seed_limit}")
    seeds = _get_seed_papers(limit=args.seed_limit)
    if not seeds:
        print("[error] 向量库中没有可用种子论文（检查 RAG 构建和 backend 服务）")
        return
    random.shuffle(seeds)

    samples: List[EvalSample] = []

    def _need_more() -> bool:
        return len(samples) < args.num

    if args.mode == "general":
        for seed in seeds:
            if not _need_more():
                break
            ss = build_general_samples_from_seed(seed)
            for s in ss:
                if not _need_more():
                    break
                samples.append(s)
    elif args.mode == "specific":
        for seed in seeds:
            if not _need_more():
                break
            s = build_specific_method_sample(seed)
            if s:
                samples.append(s)
    elif args.mode == "survey":
        for seed in seeds:
            if not _need_more():
                break
            if not _is_survey_like(seed.title):
                continue
            s = build_survey_sample(seed)
            if s:
                samples.append(s)
    else:  # mixed
        # 简单配比：general : specific : survey ≈ 2 : 1 : 1
        target_general = max(1, int(args.num * 0.5))
        target_specific = max(1, int(args.num * 0.25))
        target_survey = args.num - target_general - target_specific

        random.shuffle(seeds)
        # 先 general
        for seed in seeds:
            if len([s for s in samples if s.query_type == "general_recommendation"]) >= target_general:
                break
            ss = build_general_samples_from_seed(seed)
            for s in ss:
                if len([x for x in samples if x.query_type == "general_recommendation"]) >= target_general:
                    break
                samples.append(s)

        random.shuffle(seeds)
        # specific
        for seed in seeds:
            if len([s for s in samples if s.query_type == "specific_method"]) >= target_specific:
                break
            s = build_specific_method_sample(seed)
            if s:
                samples.append(s)

        random.shuffle(seeds)
        # survey
        for seed in seeds:
            if len([s for s in samples if s.query_type == "survey_overview"]) >= target_survey:
                break
            if not _is_survey_like(seed.title):
                continue
            s = build_survey_sample(seed)
            if s:
                samples.append(s)

    # 截断到指定数量
    samples = samples[: args.num]
    out_obj = {
        "num_samples": len(samples),
        "samples": [asdict(s) for s in samples],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"[done] Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()

