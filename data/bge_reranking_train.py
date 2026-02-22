#!/usr/bin/env python3
"""
BGE Reranker LoRA Finetune + Evaluation Comparison

在 BGE-reranker 上做 LoRA 微调，训练前后分别评估 NDCG / MRR / Recall，
并可选与 LLM 直接重排对比。

用法:
  # LoRA 微调 + 评估对比
  python data/bge_reranking_train.py --data /tmp/listwise_data/listwise_500.json

  # 加上 LLM 对比（需设 DEEPSEEK_API_KEY）
  python data/bge_reranking_train.py --data /tmp/listwise_data/listwise_500.json --eval-llm

  # 全参数微调（不用 LoRA）
  python data/bge_reranking_train.py --data /tmp/listwise_data/listwise_500.json --no-lora
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT = "./output/bge-finetuned"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 5
DEFAULT_LR = 5e-6
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
MAX_CANDIDATES = 12
PAD_DOC_PLACEHOLDER = "."


# ============================================================================
# Data loading
# ============================================================================
def _paper_to_text(paper: dict) -> str:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    return f"{title} [SEP] {abstract}"


def _user_profile_to_text(profile: dict) -> str:
    parts = []
    if profile.get("interest_field"):
        f = profile["interest_field"]
        parts.append(f"Interests: {', '.join(f) if isinstance(f, list) else f}")
    if profile.get("position"):
        parts.append(f"Position: {profile['position']}")
    if profile.get("goal"):
        parts.append(f"Goal: {profile['goal']}")
    if profile.get("professional_level"):
        parts.append(f"Level: {profile['professional_level']}")
    return "; ".join(parts) if parts else ""


def load_listwise(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj.get("samples") if isinstance(obj, dict) else obj
    if not isinstance(samples, list):
        samples = [obj]

    out = []
    for s in samples:
        q = (s.get("query") or "").strip()
        profile = s.get("user_profile") or {}
        profile_txt = _user_profile_to_text(profile)
        if profile_txt:
            q = f"{q}\n{profile_txt}"

        cands = s.get("candidate_list") or []
        labels = s.get("label") or []
        if len(labels) != len(cands):
            labels = [0.0] * len(cands)
        if len(cands) < 2:
            continue

        doc_texts = [_paper_to_text(c) for c in cands]
        valid = [i for i, d in enumerate(doc_texts) if d.strip()]
        if len(valid) < 2:
            continue
        doc_texts = [doc_texts[i] for i in valid]
        labels = [float(labels[i]) for i in valid]

        out.append({
            "query": q,
            "doc_texts": doc_texts,
            "labels": labels,
            "raw_profile": profile,
            "raw_cands": [cands[i] for i in valid],
        })
    return out


# ============================================================================
# Evaluation metrics: NDCG, MRR, Recall
# ============================================================================
def ndcg_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int = 0) -> float:
    n = len(pred_scores)
    if n < 2:
        return 0.0
    k = k if k > 0 else n
    k = min(k, n)
    pred_order = np.argsort(-pred_scores)
    ideal_order = np.argsort(-true_labels)
    dcg = sum(true_labels[pred_order[j]] / math.log2(j + 2) for j in range(k))
    idcg = sum(true_labels[ideal_order[j]] / math.log2(j + 2) for j in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(pred_scores: np.ndarray, true_labels: np.ndarray, threshold: float = 0.5) -> float:
    """MRR for one query: 1/rank of the most relevant doc in predicted order.
    MRR = (1/|Q|) * Σ 1/rank_i is computed by averaging across queries in evaluate_on_samples."""
    best_true_idx = int(np.argmax(true_labels))
    if true_labels[best_true_idx] < threshold:
        return 0.0
    pred_order = np.argsort(-pred_scores)
    rank = int(np.where(pred_order == best_true_idx)[0][0])
    return 1.0 / (rank + 1)


def recall_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int = 3, threshold: float = 0.5) -> float:
    """Recall@k: fraction of relevant docs (label >= threshold) found in top-k predicted."""
    relevant = set(np.where(true_labels >= threshold)[0])
    if not relevant:
        return 1.0
    k = min(k, len(pred_scores))
    top_k_pred = set(np.argsort(-pred_scores)[:k])
    return len(relevant & top_k_pred) / len(relevant)


def compute_all_metrics(pred_scores: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    return {
        "NDCG@3": ndcg_at_k(pred_scores, true_labels, k=3),
        "NDCG@5": ndcg_at_k(pred_scores, true_labels, k=5),
        "NDCG":   ndcg_at_k(pred_scores, true_labels, k=0),
        "MRR":    mrr(pred_scores, true_labels),
        "Recall@3": recall_at_k(pred_scores, true_labels, k=3),
        "Recall@5": recall_at_k(pred_scores, true_labels, k=5),
    }


def evaluate_on_samples(
    score_fn,
    samples: List[Dict[str, Any]],
    desc: str = "",
) -> Dict[str, float]:
    """Run score_fn(query, doc_texts) -> scores on each sample, aggregate metrics."""
    all_metrics: Dict[str, List[float]] = {}
    for i, s in enumerate(samples):
        scores = score_fn(s["query"], s["doc_texts"])
        if scores is None:
            continue
        pred = np.array(scores, dtype=float)
        true = np.array(s["labels"], dtype=float)
        m = compute_all_metrics(pred, true)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)
    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    if desc:
        logger.info("=== %s (%d samples) ===", desc, len(samples))
        for k in ["NDCG@3", "NDCG@5", "NDCG", "MRR", "Recall@3", "Recall@5"]:
            logger.info("  %-10s %.4f", k, avg.get(k, 0.0))
    return avg


# ============================================================================
# BGE scoring function (for eval)
# ============================================================================
def make_bge_score_fn(model, tokenizer, device: str, max_len: int = 512):
    def score_fn(query: str, doc_texts: List[str]) -> List[float]:
        pairs = [[query, d] for d in doc_texts]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            outputs = model(**inputs)
            scores = outputs.logits.view(-1).float().cpu().tolist()
        return scores
    return score_fn


# ============================================================================
# LLM reranking scoring function (for comparison)
# ============================================================================
def make_llm_score_fn(api_key: str, base_url: str, model_name: str):
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed, LLM eval disabled")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    def score_fn(query: str, doc_texts: List[str]) -> Optional[List[float]]:
        n = len(doc_texts)
        cands_block = ""
        for i, d in enumerate(doc_texts):
            cands_block += f"[{i+1}] {d[:300]}\n"

        prompt = (
            f"You are an expert paper ranker. Given a query and {n} candidate papers, "
            f"return a JSON object with key \"order\" containing a list of paper indices (1-based) "
            f"ranked from most to least relevant. Return ONLY valid JSON.\n\n"
            f"Query: {query}\n\nCandidates:\n{cands_block}\n"
            f"Output: {{\"order\": [...]}}"
        )
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                obj = json.loads(m.group(0))
                order = obj.get("order", [])
                scores = [0.0] * n
                for rank, idx1 in enumerate(order):
                    idx0 = int(idx1) - 1
                    if 0 <= idx0 < n:
                        scores[idx0] = float(n - rank)
                return scores
        except Exception as e:
            logger.warning("LLM rerank failed for one sample: %s", e)
        return [0.0] * n

    return score_fn


# ============================================================================
# Dataset + Collator + Trainer (same as before)
# ============================================================================
class ListwiseDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx].copy()


@dataclass
class ListwiseCollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int = 512
    max_candidates: int = MAX_CANDIDATES

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_pairs = []
        valid_masks = []
        relevance_labels = []

        for f in features:
            q = f["query"]
            doc_texts = f["doc_texts"][: self.max_candidates]
            labels = f["labels"][: self.max_candidates]

            k = len(doc_texts)
            pairs = [[q, d] for d in doc_texts]
            while len(pairs) < self.max_candidates:
                pairs.append([q, PAD_DOC_PLACEHOLDER])
                labels.append(-1e9)

            batch_pairs.extend(pairs)
            valid_masks.append([1.0] * k + [0.0] * (self.max_candidates - k))
            relevance_labels.append(labels[:k] + [0.0] * (self.max_candidates - k))

        encoded = self.tokenizer(
            batch_pairs, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        encoded["valid_mask"] = torch.tensor(valid_masks, dtype=torch.float32)
        encoded["relevance_labels"] = torch.tensor(relevance_labels, dtype=torch.float32)
        return encoded


class ListwiseTrainer(Trainer):
    def _pop_custom_fields(self, inputs):
        valid_mask = inputs.pop("valid_mask", None)
        relevance_labels = inputs.pop("relevance_labels", None)
        return valid_mask, relevance_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        valid_mask, relevance_labels = self._pop_custom_fields(inputs)

        outputs = model(**inputs)
        logits = outputs.logits
        B, K = valid_mask.size()
        scores = logits.view(B, K)

        scores_masked = scores.masked_fill(valid_mask == 0, -1e9)
        P_pred = torch.softmax(scores_masked, dim=1).clamp(min=1e-10)
        labels_masked = relevance_labels.masked_fill(valid_mask == 0, -1e9)
        P_target = torch.softmax(labels_masked, dim=1)

        ce = -(P_target * torch.log(P_pred)).sum(dim=1)
        n_valid = valid_mask.sum(dim=1).clamp(min=1)
        loss = (ce / n_valid).mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        valid_mask, relevance_labels = self._pop_custom_fields(inputs)
        loss = None
        with torch.no_grad():
            if valid_mask is not None and relevance_labels is not None:
                outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
                logits = outputs.logits
                B, K = valid_mask.size()
                scores = logits.view(B, K)
                vm = valid_mask.to(scores.device)
                rl = relevance_labels.to(scores.device)
                scores_masked = scores.masked_fill(vm == 0, -1e9)
                P_pred = torch.softmax(scores_masked, dim=1).clamp(min=1e-10)
                labels_masked = rl.masked_fill(vm == 0, -1e9)
                P_target = torch.softmax(labels_masked, dim=1)
                ce = -(P_target * torch.log(P_pred)).sum(dim=1)
                n_valid = vm.sum(dim=1).clamp(min=1)
                loss = (ce / n_valid).mean()
                return (loss, logits, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


# ============================================================================
# LoRA setup
# ============================================================================
def setup_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.1):
    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ============================================================================
# Comparison table
# ============================================================================
def print_comparison(results: Dict[str, Dict[str, float]]):
    metrics = ["NDCG@3", "NDCG@5", "NDCG", "MRR", "Recall@3", "Recall@5"]
    names = list(results.keys())

    header = f"{'Metric':<12}" + "".join(f"{n:<20}" for n in names)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(f"{'':>12}  EVALUATION COMPARISON")
    print(sep)
    print(header)
    print(sep)
    for m in metrics:
        row = f"{m:<12}"
        vals = [results[n].get(m, 0.0) for n in names]
        best = max(vals)
        for v in vals:
            marker = " *" if v == best and len(names) > 1 else "  "
            row += f"{v:<18.4f}{marker}"
        print(row)
    print(sep)
    print("(* = best)\n")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="BGE Reranker LoRA Finetune + Evaluation")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Held-out test set ratio for final comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--no-lora", action="store_true", help="全参数微调（不用 LoRA）")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument("--eval-llm", action="store_true",
                        help="同时用 LLM 重排做对比评估")
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-base-url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat")
    parser.add_argument("--llm-eval-n", type=int, default=50,
                        help="LLM 评估最多用多少个 test samples（API 调用贵/慢）")
    parser.add_argument("--wandb", type=str, default="")
    args = parser.parse_args()

    # ---- Load data ----
    logger.info("Loading data from %s", args.data)
    samples = load_listwise(args.data)
    if not samples:
        raise SystemExit("No valid samples found.")
    logger.info("Total valid samples: %d", len(samples))

    random.seed(args.seed)
    random.shuffle(samples)

    n_test = max(1, int(len(samples) * args.test_ratio))
    n_val = max(1, int(len(samples) * args.val_ratio))
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    logger.info("Split: train=%d, val=%d, test=%d", len(train_samples), len(val_samples), len(test_samples))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- 1) Evaluate BASE model (before finetune) ----
    logger.info("Loading base model: %s", args.model)
    base_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    base_model.eval()

    base_score_fn = make_bge_score_fn(base_model, tokenizer, device, args.max_len)
    base_metrics = evaluate_on_samples(base_score_fn, test_samples, desc="Base BGE (before finetune)")
    del base_model
    torch.cuda.empty_cache()

    # ---- 2) LoRA finetune ----
    logger.info("Loading model for finetuning...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    use_lora = not args.no_lora
    if use_lora:
        logger.info("Applying LoRA (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
        model = setup_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    else:
        logger.info("Full parameter finetuning (no LoRA)")

    use_fp16 = not args.no_fp16 and not args.bf16 and torch.cuda.is_available()
    use_bf16 = args.bf16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=["wandb"] if args.wandb else "none",
        remove_unused_columns=False,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb

    collator = ListwiseCollator(tokenizer, max_len=args.max_len, max_candidates=args.max_candidates)
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    eff_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_samples) // eff_batch) * args.epochs
    logger.info("Effective batch=%d, ~%d steps, lr=%.1e, %s",
                eff_batch, total_steps, args.lr, "LoRA" if use_lora else "full")

    trainer = ListwiseTrainer(
        model=model,
        args=training_args,
        train_dataset=ListwiseDataset(train_samples),
        eval_dataset=ListwiseDataset(val_samples),
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Starting finetune...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info("Finetune done in %.1f min", elapsed / 60)

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    if use_lora:
        model.save_pretrained(args.output)
    logger.info("Model saved to %s", args.output)

    # ---- 3) Evaluate FINETUNED model ----
    model.eval()
    model.to(device)
    ft_score_fn = make_bge_score_fn(model, tokenizer, device, args.max_len)
    ft_metrics = evaluate_on_samples(ft_score_fn, test_samples, desc="BGE + LoRA (after finetune)")

    # ---- 4) Optionally evaluate LLM reranker ----
    results = {
        "Base BGE": base_metrics,
        "BGE+LoRA FT": ft_metrics,
    }

    if args.eval_llm:
        api_key = args.llm_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("No DEEPSEEK_API_KEY set, skipping LLM eval")
        else:
            llm_fn = make_llm_score_fn(api_key, args.llm_base_url, args.llm_model)
            if llm_fn:
                llm_test = test_samples[:args.llm_eval_n]
                logger.info("Evaluating LLM reranker (%s) on %d samples...", args.llm_model, len(llm_test))
                llm_metrics = evaluate_on_samples(llm_fn, llm_test, desc=f"LLM ({args.llm_model})")
                results[f"LLM ({args.llm_model})"] = llm_metrics

                base_llm = evaluate_on_samples(base_score_fn if 'base_score_fn' in dir() else ft_score_fn,
                                                llm_test, desc="")
                results["Base BGE (same subset)"] = base_llm
                ft_llm = evaluate_on_samples(ft_score_fn, llm_test, desc="")
                results["BGE+LoRA FT (same subset)"] = ft_llm

    # ---- 5) Print comparison table ----
    print_comparison(results)


if __name__ == "__main__":
    main()
