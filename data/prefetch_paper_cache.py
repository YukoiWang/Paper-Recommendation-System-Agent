# -*- coding: utf-8 -*-
"""
预拉取引用列表和论文元数据到本地缓存，供 create_listwise2 使用。

用法:
  cd /path/to/Paper-Recommendation-System-Agent
  source ~/paper-rec-venv/bin/activate
  python data/prefetch_paper_cache.py -o ~/listwise_cache -n 10000

完成后运行 create_listwise2 时加 --cache-dir ~/listwise_cache
"""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PARENT = _SCRIPT_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# Stub before ChromaDB
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

class _PosthogStub:
    disabled = True
    project_api_key = ""
    @staticmethod
    def capture(*args, **kwargs):
        pass
sys.modules["posthog"] = _PosthogStub()

from paper_cache import PaperCache
from rag_listwise_helpers import get_seed_paper_ids_from_rag, get_paper_from_rag


def main():
    parser = argparse.ArgumentParser(
        description="预拉取引用和元数据到本地缓存，供 create_listwise2 加速"
    )
    parser.add_argument("-o", "--out", type=str, default=os.path.expanduser("~/listwise_cache"),
                        help="缓存目录 (default: ~/listwise_cache)")
    parser.add_argument("-n", "--num", type=int, default=10000,
                        help="最多处理多少篇 RAG 论文 (default: 10000)")
    parser.add_argument("--chroma-path", type=str, default="",
                        help="ChromaDB 路径 (默认用 CHROMA_PERSIST_DIR)")
    args = parser.parse_args()

    cache = PaperCache(args.out)
    chroma_path = args.chroma_path or os.getenv("CHROMA_PERSIST_DIR") or os.path.expanduser("~/chroma_db")

    print(f"[prefetch] 缓存目录: {args.out}")
    print(f"[prefetch] 目标: 最多 {args.num} 篇 RAG 论文")

    paper_ids = get_seed_paper_ids_from_rag(
        limit=args.num + 500, chroma_path=chroma_path or None
    )
    paper_ids = paper_ids[: args.num]
    if not paper_ids:
        print("No paper IDs from RAG. Ensure ChromaDB is populated.")
        cache.close()
        return

    print(f"[prefetch] 从 RAG 获取 {len(paper_ids)} 篇论文 ID")

    # 延迟导入 create_listwise2 的 API（避免循环依赖、减少启动开销）
    from create_listwise2 import (
        get_reference_list_by_paper_id,
        get_work,
        get_paper_from_s2,
        work_to_candidate,
        _normalize_paper_id,
        _is_arxiv_id,
    )

    def _norm(s: str) -> str:
        return _normalize_paper_id(s) if s else ""

    all_ref_ids = set()
    done_refs = 0
    done_meta = 0

    # 1) 拉取引用列表
    for i, pid in enumerate(paper_ids):
        raw_id = _norm(pid)
        if cache.has_refs(raw_id):
            continue
        refs = get_reference_list_by_paper_id(raw_id)
        if refs:
            cache.set_refs(raw_id, refs)
            all_ref_ids.update(refs)
            done_refs += 1
        if (i + 1) % 100 == 0:
            print(f"  refs: {i+1}/{len(paper_ids)}, 已缓存 {cache.refs_count()} 篇")

    print(f"[prefetch] 引用列表完成: {cache.refs_count()} 篇, 共 {len(all_ref_ids)} 个唯一引用 ID")

    # 2) 拉取引用论文的元数据（仅当不在 RAG 时）
    rag_ids = set()
    if get_paper_from_rag:
        for pid in paper_ids:
            nid = _norm(pid)
            p = get_paper_from_rag(nid)
            if p:
                rag_ids.add(nid)

    to_fetch = []
    seen = set()
    for ref_id in all_ref_ids:
        rid = _norm(ref_id)
        lookup = rid[1:] if rid and rid.upper().startswith("W") and _is_arxiv_id(rid[1:]) else rid
        if lookup in rag_ids:
            continue
        if cache.has_meta(rid) or cache.has_meta(lookup):
            continue
        if rid not in seen:
            seen.add(rid)
            to_fetch.append(rid)

    print(f"[prefetch] 需拉取元数据: {len(to_fetch)} 篇 (排除 RAG 已有)")

    for i, rid in enumerate(to_fetch):
        if cache.has_meta(rid):
            continue
        w = get_work(rid)
        if w:
            c = work_to_candidate(w)
            if c:
                key = _norm(c.get("paper_id", rid))
                if key:
                    cache.set_meta(key, c)
                    done_meta += 1
        else:
            c = get_paper_from_s2(rid)
            if c:
                key = _norm(c.get("paper_id", rid))
                if key:
                    cache.set_meta(key, c)
                    done_meta += 1
        if (i + 1) % 50 == 0:
            print(f"  meta: {i+1}/{len(to_fetch)}, 已缓存 {cache.meta_count()} 篇")

    n_refs, n_meta = cache.refs_count(), cache.meta_count()
    cache.close()
    print(f"[prefetch] 完成. refs: {n_refs}, meta: {n_meta}")
    print(f"  使用: python data/create_listwise2.py --cache-dir {args.out} ...")


if __name__ == "__main__":
    main()
