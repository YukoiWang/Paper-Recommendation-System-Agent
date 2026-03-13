#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end RAG 评估脚本（完整系统：检索 + 重排 + 生成）：

1. 读取 data/create_rag_eval_dataset.py 生成的评估集 JSON
2. 对每个样本，用当前 LangGraph RAG 系统跑通整条 pipeline 得到：
   - 检索+重排后的 final_papers（用于 Recall@k）
   - 生成回答与 contexts（用于 Ragas）
3. 检索指标：Recall@5/10/15/20（与 ground_truth_paper_ids 对比）
4. 生成指标（ragas）：answer_relevancy, context_precision, context_recall, faithfulness
5. 按 query_type 划分子集，结果写入 --metrics-out

使用示例：

  python3.10 data/eval_rag_with_system_ragas.py \
    --eval-json /tmp/rag_eval_150.json \
    --chromadb-path /tmp/chroma_db \
    --api-key "$DEEPSEEK_API_KEY" \
    --metrics-out /tmp/rag_eval_ragas_metrics.json

可选：--recall-ks 5,10,15,20  --top-k 20（top_k 会至少为 max(recall_ks)）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pysqlite3 hack for Chroma sqlite>=3.35
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv  # type: ignore

load_dotenv(ROOT / ".env")


def _ensure_ragas():
    """Import ragas & datasets, and prepare metrics + evaluator models.

    - metrics/evaluate/Dataset: 来自 ragas 本身
    - evaluator_llm: 用 DeepSeek Chat（通过 openai 兼容客户端 + ragas.llm_factory）
    - evaluator_embeddings: 用本地 HuggingFace sentence-transformers（不依赖 OpenAI）
    """

    try:
        from ragas import evaluate  # type: ignore
        # 使用 ragas.metrics 中已经初始化好的 Metric 实例（虽然有弃用警告，但兼容 0.4.x）
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print(
            "ragas 未安装。\n"
            "请先运行：\n"
            "  pip install \"ragas[eval]\" \"datasets>=2.14.0\"",
            file=sys.stderr,
        )
        raise

    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        print(
            "datasets 未安装。\n"
            "请先运行：\n"
            "  pip install \"datasets>=2.14.0\"",
            file=sys.stderr,
        )
        raise

    metrics = [answer_relevancy, context_precision, context_recall, faithfulness]

    # --- 配置 ragas 评估 LLM（DeepSeek）和 Embeddings（HuggingFace，本地） ---
    evaluator_llm = None
    evaluator_embeddings = None

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            from openai import OpenAI  # type: ignore
            from ragas.llms import llm_factory  # type: ignore
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            ds_base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            client = OpenAI(api_key=ds_key, base_url=ds_base)

            # 1) 评估 LLM：DeepSeek Chat
            max_tokens = int(os.environ.get("RAGAS_EVAL_MAX_TOKENS", "4096"))
            evaluator_llm = llm_factory(
                model=os.environ.get("RAGAS_DEEPSEEK_MODEL", "deepseek-chat"),
                client=client,
                temperature=0.0,
                max_tokens=max_tokens,
            )

            # 2) 评估 Embeddings：本地 HuggingFace sentence-transformers
            #    不依赖 OpenAI / DeepSeek，适合作为通用嵌入后端。
            hf_model = os.environ.get(
                "RAGAS_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            evaluator_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=hf_model)
            )
        except Exception as e:
            print(
                f"Warning: 初始化 ragas DeepSeek LLM / HF Embeddings 失败，将退回默认配置: {e}",
                file=sys.stderr,
            )
            evaluator_llm = None
            evaluator_embeddings = None

    return evaluate, Dataset, metrics, evaluator_llm, evaluator_embeddings


def _load_eval_samples(path: Path) -> List[Dict[str, Any]]:
    """Load eval samples from create_rag_eval_dataset.py output."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported eval json format: {path}")


def _get_ground_truth_paper_ids(sample: Dict[str, Any]) -> List[str]:
    """Extract ground truth paper IDs from eval sample (for Recall@k)."""
    ids = sample.get("ground_truth_paper_ids")
    if isinstance(ids, list) and ids:
        return [str(x).strip() for x in ids if x]
    contexts = sample.get("ground_truth_contexts") or []
    out: List[str] = []
    for c in contexts:
        if isinstance(c, dict):
            pid = c.get("paper_id") or c.get("id")
            if pid:
                out.append(str(pid).strip())
    return out


def _recall_at_k(retrieved_ids: List[str], gt_ids: List[str], k: int) -> float:
    """Recall@k: |retrieved[:k] ∩ gt| / |gt|. Returns 0.0 if gt is empty."""
    if not gt_ids:
        return 0.0
    gt_set = set(gt_ids)
    hit = len(gt_set & set(retrieved_ids[:k]))
    return hit / len(gt_set)


def _build_contexts_from_papers(
    papers: List[Any],
    max_papers: int = 8,
    max_abstract_chars: int = 300,
) -> List[str]:
    """Convert final_papers (Paper objects) into ragas contexts."""
    out: List[str] = []
    for p in papers[:max_papers]:
        title = getattr(p, "title", "") or ""
        abstract = getattr(p, "abstract", "") or ""
        if max_abstract_chars and abstract:
            abstract = abstract[:max_abstract_chars]
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        text = "\n".join(parts).strip()
        if text:
            out.append(text)
    return out


async def _ainvoke_workflow(app, state: Dict[str, Any], timeout: float = 90.0) -> Dict[str, Any]:
    """Async helper with timeout for app.ainvoke."""
    async def _invoke():
        return await asyncio.wait_for(app.ainvoke(state), timeout=timeout)

    return await _invoke()


def _run_workflow_for_query(
    app,
    user_id: str,
    query: str,
    top_k: int,
    history: List[Dict[str, Any]] | None = None,
    cited_papers: Dict[str, Any] | None = None,
    max_context_papers: int | None = None,
    max_context_abstract_chars: int | None = None,
) -> Dict[str, Any]:
    """Single-turn invocation of LangGraph workflow, modeled after run_workflow.run_single.
    When max_context_papers/max_context_abstract_chars are set, QA uses only that
    many papers and abstract length (same as Ragas context) so answer and Ragas align."""
    from agent.models import UserProfile  # type: ignore

    profile = UserProfile(user_id=user_id, interest_text="machine learning")
    state: Dict[str, Any] = {
        "user_id": profile.user_id,
        "user_profile": profile,
        "user_query": query,
        "user_feedback": "",
        "is_daily_rec": False,
        "top_k": top_k,
        "online_offline_fusion_ratio": 0.5,
        "history": history or [],
        "cited_papers": cited_papers or {},
        "evaluation_mode": True,
    }
    if max_context_papers is not None:
        state["max_context_papers"] = max_context_papers
    if max_context_abstract_chars is not None:
        state["max_context_abstract_chars"] = max_context_abstract_chars
    result = asyncio.run(_ainvoke_workflow(app, state))
    return result


def _build_workflow_app(
    api_key: str,
    top_k: int,
    use_chromadb: bool,
    chromadb_path: str,
    embedding_model: str,
):
    """Reuse langgraph_agents.run_workflow._build_agents to construct app & retrieval."""
    from langgraph_agents.run_workflow import _build_agents  # type: ignore

    app, retrieval = _build_agents(
        api_key=api_key,
        top_k=top_k,
        use_chromadb=use_chromadb,
        chromadb_path=chromadb_path,
        embedding_model=embedding_model,
    )
    return app, retrieval


def _build_ragas_rows_for_subset(
    app,
    samples: List[Dict[str, Any]],
    subset_name: str,
    top_k: int,
    max_context_papers: int,
    context_abstract_chars: int,
    max_answer_chars: int,
    recall_ks: List[int],
) -> Tuple[
    List[str],
    List[str],
    List[List[str]],
    List[str],
    Dict[int, List[float]],
]:
    """Run workflow for each sample in subset and collect ragas fields + retrieval Recall@k.

    Returns:
      questions, system_answers, contexts_list, ground_truths, recall_at_k_lists
      where recall_at_k_lists[k] = [recall@k for each sample with non-empty gt_ids].
    """
    questions: List[str] = []
    system_answers: List[str] = []
    ground_truths: List[str] = []
    contexts_list: List[List[str]] = []
    recall_at_k_lists: Dict[int, List[float]] = {k: [] for k in recall_ks}

    for idx, s in enumerate(samples, start=1):
        q = (s.get("query") or "").strip()
        gt_answer = (s.get("answer") or "").strip()
        if not q or not gt_answer:
            continue

        gt_paper_ids = _get_ground_truth_paper_ids(s)
        print(f"[{subset_name}] {idx}/{len(samples)}  query={q[:60]!r}")
        try:
            result = _run_workflow_for_query(
                app,
                user_id=f"eval-{subset_name}",
                query=q,
                top_k=top_k,
                max_context_papers=max_context_papers,
                max_context_abstract_chars=context_abstract_chars,
            )
        except Exception as e:
            print(f"  -> workflow error, skip: {e}")
            continue

        sys_answer = (result.get("response") or "").strip()
        if max_answer_chars and len(sys_answer) > max_answer_chars:
            sys_answer = sys_answer[:max_answer_chars].rstrip() + "..."
        final_papers = result.get("final_papers") or []
        retrieved_ids = [
            getattr(p, "paper_id", "") or str(getattr(p, "id", ""))
            for p in final_papers
            if getattr(p, "paper_id", None) or getattr(p, "id", None)
        ]
        if not retrieved_ids and final_papers:
            retrieved_ids = [getattr(p, "paper_id", "") for p in final_papers]

        # Recall@k (retrieval+rerank of the full system)
        if gt_paper_ids:
            for k in recall_ks:
                recall_at_k_lists[k].append(_recall_at_k(retrieved_ids, gt_paper_ids, k))

        ctxs = _build_contexts_from_papers(
            final_papers,
            max_papers=max_context_papers,
            max_abstract_chars=context_abstract_chars,
        )

        if not sys_answer or not ctxs:
            print("  -> empty response or contexts, skip")
            continue

        questions.append(q)
        system_answers.append(sys_answer)
        ground_truths.append(gt_answer)
        contexts_list.append(ctxs)

    return questions, system_answers, contexts_list, ground_truths, recall_at_k_lists


def main() -> int:
    parser = argparse.ArgumentParser(description="Run system RAG + ragas evaluation on synthetic dataset")
    parser.add_argument(
        "--eval-json",
        type=str,
        required=True,
        help="Path to eval dataset JSON from create_rag_eval_dataset.py",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="LLM API key for LangGraph workflow (default: DEEPSEEK_API_KEY env)",
    )
    parser.add_argument(
        "--chromadb-path",
        type=str,
        default=os.environ.get("CHROMA_PERSIST_DIR", "/tmp/chroma_db"),
        help="Path to ChromaDB used by retrieval agent",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Workflow top_k (retrieval+rerank). Should be >= max(recall-ks) for Recall@k.",
    )
    parser.add_argument(
        "--recall-ks",
        type=str,
        default="5,10,15,20",
        help="Comma-separated k values for Recall@k (e.g. 5,10,15,20). Full system retrieval+rerank.",
    )
    parser.add_argument(
        "--max-context-papers",
        type=int,
        default=4,
        help="Max final_papers to turn into ragas contexts per sample",
    )
    parser.add_argument(
        "--context-abstract-chars",
        type=int,
        default=300,
        help="Max abstract chars per context chunk (to control ragas prompt length)",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=1200,
        help="Max chars of system answer used for ragas evaluation (to avoid LLM output truncation)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total samples used from eval-json (0 = use all)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name used for Chroma queries",
    )
    parser.add_argument(
        "--bm25-json",
        type=str,
        default=str(ROOT / "data" / "papers_62k_for_bm25.json"),
        help="Optional JSON file used to initialize BM25 for hybrid retrieval",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to save aggregated ragas metrics as JSON",
    )
    parser.add_argument(
        "--save-ragas-json",
        type=str,
        default="",
        help="Optional path to save constructed ragas dataset (list of rows)",
    )
    args = parser.parse_args()

    # Parse Recall@k values and ensure top_k >= max(recall_ks)
    recall_ks: List[int] = []
    for part in (args.recall_ks or "5,10,15,20").strip().split(","):
        part = part.strip()
        if part:
            try:
                recall_ks.append(int(part))
            except ValueError:
                pass
    if not recall_ks:
        recall_ks = [5, 10, 15, 20]
    recall_ks = sorted(set(recall_ks))
    if args.top_k < max(recall_ks):
        args.top_k = max(recall_ks)
        print(f"Adjusted top_k to {args.top_k} (>= max recall-ks {recall_ks})")

    eval_path = Path(args.eval_json).expanduser()
    if not eval_path.is_file():
        print(f"eval-json not found: {eval_path}", file=sys.stderr)
        return 1
    if not args.api_key:
        print("Error: --api-key 未设置，且环境变量 DEEPSEEK_API_KEY 为空。", file=sys.stderr)
        return 1

    samples = _load_eval_samples(eval_path)
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        print("No samples loaded from eval-json.", file=sys.stderr)
        return 1

    print("Building workflow app...")
    app, retrieval = _build_workflow_app(
        api_key=args.api_key,
        top_k=args.top_k,
        use_chromadb=True,
        chromadb_path=args.chromadb_path,
        embedding_model=args.embedding_model,
    )
    print(f"Using ChromaDB at {args.chromadb_path} (index_size={getattr(retrieval, 'index_size', 'N/A')})")

    bm25_json = (args.bm25_json or "").strip()
    if bm25_json and hasattr(retrieval, "init_bm25_from_json"):
        bm25_path = Path(bm25_json).expanduser()
        if bm25_path.is_file():
            try:
                n_bm25 = retrieval.init_bm25_from_json(str(bm25_path))
                print(f"Initialized BM25 from {bm25_path} ({n_bm25} docs)")
            except Exception as e:
                print(f"[warn] BM25 init failed from {bm25_path}: {e}", file=sys.stderr)
        else:
            print(f"[warn] BM25 JSON not found, skip hybrid retrieval: {bm25_path}", file=sys.stderr)

    # Prepare ragas (metrics + evaluator models)
    evaluate, Dataset, metrics, evaluator_llm, evaluator_embeddings = _ensure_ragas()

    # Split subsets by query_type
    subsets: Dict[str, List[Dict[str, Any]]] = {
        "general": [s for s in samples if s.get("query_type") == "general_recommendation"],
        "specific": [s for s in samples if s.get("query_type") == "specific_method"],
        "survey": [s for s in samples if s.get("query_type") == "survey_overview"],
        "mixed": list(samples),
    }

    all_metrics: Dict[str, Any] = {}
    ragas_rows_all: List[Dict[str, Any]] = []

    for name, subs in subsets.items():
        if not subs:
            print(f"[warn] subset '{name}' is empty, skip.", file=sys.stderr)
            continue

        print(f"\n==================== {name.upper()} ====================")
        print(f"#eval samples = {len(subs)}")

        questions, sys_answers, contexts_list, gts, recall_at_k_lists = _build_ragas_rows_for_subset(
            app,
            subs,
            subset_name=name,
            top_k=args.top_k,
            max_context_papers=args.max_context_papers,
            context_abstract_chars=args.context_abstract_chars,
            max_answer_chars=args.max_answer_chars,
            recall_ks=recall_ks,
        )

        # Retrieval metrics (full system: retrieve + rerank)
        retrieval_metrics: Dict[str, float] = {}
        for k in recall_ks:
            vals = recall_at_k_lists.get(k) or []
            if vals:
                retrieval_metrics[f"recall@{k}"] = sum(vals) / len(vals)
        if retrieval_metrics:
            print(f"Retrieval (recall): {retrieval_metrics}")

        if not questions:
            print(f"[warn] subset '{name}' produced no valid rows, skip ragas.", file=sys.stderr)
            if retrieval_metrics:
                all_metrics[name] = {"retrieval": retrieval_metrics}
            continue

        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": sys_answers,
                "contexts": contexts_list,
                "ground_truth": gts,
                "reference": gts,
            }
        )

        # Also accumulate rows for optional save
        for q, a, ctxs, gt in zip(questions, sys_answers, contexts_list, gts):
            ragas_rows_all.append(
                {
                    "subset": name,
                    "question": q,
                    "answer": a,
                    "contexts": ctxs,
                    "ground_truth": gt,
                    "reference": gt,
                }
            )

        # 显式把 DeepSeek LLM + HF Embeddings 传给 ragas，
        # 避免其内部再去默认 new OpenAI()。
        kwargs: Dict[str, Any] = {"metrics": metrics}
        if evaluator_llm is not None:
            kwargs["llm"] = evaluator_llm
        if evaluator_embeddings is not None:
            kwargs["embeddings"] = evaluator_embeddings

        res = evaluate(ds, **kwargs)
        print(res)

        subset_repr: Any = None
        try:
            if hasattr(res, "to_pandas"):
                df = res.to_pandas()
                subset_repr = {
                    "metrics": df.to_dict(orient="records")[0] if not df.empty else {},
                    "retrieval": retrieval_metrics,
                }
            else:
                subset_repr = {"repr": repr(res), "retrieval": retrieval_metrics}
        except Exception:
            subset_repr = {"repr": repr(res), "retrieval": retrieval_metrics}

        all_metrics[name] = subset_repr

    if args.metrics_out:
        out_path = Path(args.metrics_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved aggregated metrics to {out_path}")

    if args.save_ragas_json:
        ds_path = Path(args.save_ragas_json).expanduser()
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        with ds_path.open("w", encoding="utf-8") as f:
            json.dump(ragas_rows_all, f, ensure_ascii=False, indent=2)
        print(f"Saved ragas dataset rows to {ds_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

