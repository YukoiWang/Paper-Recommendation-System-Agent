#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 评估脚本（使用 ragas）

数据来源：data/create_rag_eval_dataset.py 生成的 JSON：
{
  "num_samples": 150,
  "samples": [
    {
      "query_type": "general_recommendation" | "specific_method" | "survey_overview",
      "query_subtype": "fuzzy" | "noisy" | "normal" | "complex" | null,
      "query": "...",
      "seed_paper_id": "...",
      "seed_paper_title": "...",
      "seed_paper_abstract": "...",
      "ground_truth_paper_ids": ["...", ...],
      "ground_truth_contexts": [
        {"paper_id": "...", "title": "...", "abstract": "...", "authors": [...]},
        ...
      ],
      "context_summary": "...",
      "answer": "..."
    },
    ...
  ]
}

本脚本会：
- 从该 JSON 中构造 ragas 需要的 dataset：
  - question: query
  - answer:   answer  （数据集里生成的理想回答）
  - contexts: 从 ground_truth_contexts 拼出的文档片段
- 分别在 4 个子集上评估：
  - general:  query_type == "general_recommendation"
  - specific: query_type == "specific_method"
  - survey:   query_type == "survey_overview"
  - mixed:    全部样本

运行示例：
  python data/eval_rag_ragas.py --dataset /tmp/rag_eval_150.json

注意：
- 需要先安装依赖：
    pip install "ragas[eval]" "datasets>=2.14.0"
- ragas 内部会调用 LLM（默认 OpenAI），需配置好对应的 API Key。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _ensure_ragas():
    """Import ragas & datasets, with友好错误提示。"""
    try:
        from ragas import evaluate  # type: ignore
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
    return evaluate, Dataset, metrics


def _build_contexts(
    gt_contexts: List[Dict[str, Any]],
    max_papers: int = 10,
    max_abstract_chars: int = 800,
) -> List[str]:
    """把 ground_truth_contexts 转成 ragas 的 contexts（若干段纯文本）。"""
    contexts: List[str] = []
    for p in gt_contexts[:max_papers]:
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        if max_abstract_chars and abstract:
            abstract = abstract[:max_abstract_chars]
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        text = "\n".join(parts).strip()
        if text:
            contexts.append(text)
    return contexts


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dataset json format in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG using ragas on synthetic eval dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to rag eval dataset JSON (from create_rag_eval_dataset.py)",
    )
    parser.add_argument(
        "--max-papers-per-sample",
        type=int,
        default=10,
        help="Max ground-truth papers to use as contexts for each sample (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save aggregated metrics as JSON",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_file():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    samples = _load_samples(dataset_path)
    if not samples:
        print("No samples found in dataset.", file=sys.stderr)
        return 1

    evaluate, Dataset, metrics = _ensure_ragas()

    # 按 query_type 划分子集
    subsets: Dict[str, List[Dict[str, Any]]] = {
        "general": [s for s in samples if s.get("query_type") == "general_recommendation"],
        "specific": [s for s in samples if s.get("query_type") == "specific_method"],
        "survey": [s for s in samples if s.get("query_type") == "survey_overview"],
        "mixed": list(samples),
    }

    all_results: Dict[str, Any] = {}

    for name, subs in subsets.items():
        if not subs:
            print(f"[warn] subset '{name}' is empty, skip.", file=sys.stderr)
            continue

        print(f"\n==================== {name.upper()} ====================")
        print(f"#samples = {len(subs)}")

        # 构造 ragas dataset
        questions: List[str] = []
        answers: List[str] = []
        contexts_list: List[List[str]] = []

        for s in subs:
            q = (s.get("query") or "").strip()
            a = (s.get("answer") or "").strip()
            gt_ctxs = s.get("ground_truth_contexts") or []
            ctx_texts = _build_contexts(
                gt_ctxs,
                max_papers=args.max_papers_per_sample,
            )
            if not q or not a or not ctx_texts:
                # 跳过信息不全的样本
                continue
            questions.append(q)
            answers.append(a)
            contexts_list.append(ctx_texts)

        if not questions:
            print(f"[warn] subset '{name}' has no valid rows after filtering, skip.", file=sys.stderr)
            continue

        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
            }
        )

        # 运行 ragas 评估
        res = evaluate(ds, metrics=metrics)

        # 直接打印 ragas 的结果对象（不同版本返回类型略有差异，统一用 repr）
        print(res)

        # 尝试提取为简单 dict 方便写出
        result_repr: Any = None
        try:
            # ragas >=0.1.4: 有 .to_pandas()
            if hasattr(res, "to_pandas"):
                df = res.to_pandas()
                result_repr = {
                    "metrics": df.to_dict(orient="records")[0] if not df.empty else {},
                }
            else:
                # fallback: 就存 repr
                result_repr = {"repr": repr(res)}
        except Exception:
            result_repr = {"repr": repr(res)}

        all_results[name] = result_repr

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved aggregated metrics to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

