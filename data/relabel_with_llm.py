#!/usr/bin/env python3
"""Re-label existing listwise data using LLM scoring (distillation).

Reads a listwise JSON file, sends each sample to an LLM for relevance scoring,
and writes a new JSON with LLM-generated labels.

Usage:
    DEEPSEEK_API_KEY=sk-xxx python -u data/relabel_with_llm.py \
        --input ~/listwise_data/listwise_500.json \
        --output ~/listwise_data/listwise_llm_500.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCORING_SYSTEM = """You are an expert academic paper recommender.
Given a seed paper and a list of candidate papers, score each candidate's relevance to the seed paper.

Scoring criteria:
- Topic similarity (same research area, related problems)
- Methodological relevance (similar or complementary techniques)
- Citation-worthiness (would the seed paper's author likely cite this?)
- Novelty & utility (does this candidate offer useful insights for the seed paper's research direction?)

Output STRICT JSON only, no markdown, no extra text."""

SCORING_USER = """## Seed paper
- Title: {seed_title}
- Abstract: {seed_abstract}

## Candidate papers
{candidates_block}

## Task
Score each candidate's relevance to the seed paper on a scale of 0.0 to 1.0:
- 1.0 = highly relevant, directly related work
- 0.7-0.9 = clearly relevant, same field/topic
- 0.4-0.6 = somewhat relevant, tangentially related
- 0.1-0.3 = weakly relevant, different field but minor connection
- 0.0 = irrelevant

## Output JSON schema (STRICT)
{{
  "scores": [0.85, 0.72, 0.31, ...]  // one float per candidate, same order as input
}}

Rules:
- "scores" array length MUST equal the number of candidates ({n_cands}).
- Each score must be a float between 0.0 and 1.0.
- Use the FULL range. Not all candidates should be high or low.
- Base scores ONLY on the provided title/abstract content."""


def build_candidates_block(candidates: List[Dict[str, Any]], max_abs: int = 400) -> str:
    parts = []
    for i, c in enumerate(candidates):
        abstract = (c.get("abstract") or "")[:max_abs]
        if len(c.get("abstract") or "") > max_abs:
            abstract += "..."
        parts.append(
            f"### Candidate {i+1}\n"
            f"- Title: {c['title']}\n"
            f"- Abstract: {abstract}"
        )
    return "\n\n".join(parts)


def score_one_sample(
    client: OpenAI,
    sample: Dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> List[float] | None:
    seed = sample["current_paper"]
    cands = sample["candidate_list"]
    n = len(cands)

    messages = [
        {"role": "system", "content": SCORING_SYSTEM},
        {
            "role": "user",
            "content": SCORING_USER.format(
                seed_title=seed["title"],
                seed_abstract=(seed.get("abstract") or "")[:600],
                candidates_block=build_candidates_block(cands),
                n_cands=n,
            ),
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()

            # Try parsing JSON
            text_clean = text
            if text_clean.startswith("```"):
                text_clean = re.sub(r"^```\w*\n?", "", text_clean)
                text_clean = re.sub(r"\n?```$", "", text_clean)

            obj = json.loads(text_clean)
            scores = obj.get("scores", [])

            if len(scores) != n:
                log.warning("Expected %d scores, got %d. Attempt %d", n, len(scores), attempt + 1)
                if attempt < max_retries - 1:
                    continue
                # Pad or truncate
                while len(scores) < n:
                    scores.append(0.3)
                scores = scores[:n]

            return [max(0.0, min(1.0, float(s))) for s in scores]

        except json.JSONDecodeError:
            # Try to extract numbers from text
            nums = re.findall(r"0?\.\d+|1\.0|0\.0", text)
            if len(nums) >= n:
                return [max(0.0, min(1.0, float(x))) for x in nums[:n]]
            log.warning("JSON parse failed, attempt %d: %s", attempt + 1, text[:200])
        except Exception as e:
            log.warning("API error attempt %d: %s", attempt + 1, e)
            time.sleep(2 ** attempt)

    return None


def main():
    parser = argparse.ArgumentParser(description="Re-label listwise data with LLM scores")
    parser.add_argument("--input", required=True, help="Input listwise JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--api-key", default=None, help="LLM API key (or DEEPSEEK_API_KEY env)")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL")
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API workers")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or set DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    log.info("Loading %s", args.input)
    with open(args.input) as f:
        data = json.load(f)

    samples = data["samples"]
    log.info("Loaded %d samples, %d candidates each", len(samples), len(samples[0]["candidate_list"]))

    new_samples = [dict(s) for s in samples]
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(score_one_sample, client, s, args.model): i
            for i, s in enumerate(samples)
        }
        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            scores = future.result()
            done_count += 1

            if scores is None:
                failed += 1
                log.warning("[%d/%d] Sample %d FAILED, keeping original labels",
                            done_count, len(samples), idx)
            else:
                new_samples[idx]["label"] = scores
                if done_count % 50 == 0 or done_count == len(samples):
                    elapsed = time.time() - t0
                    log.info("[%d/%d] %.0fs elapsed, ~%.0fs remaining",
                             done_count, len(samples), elapsed,
                             elapsed / done_count * (len(samples) - done_count))

    elapsed = time.time() - t0
    log.info("Done in %.1f min. Failed: %d/%d", elapsed / 60, failed, len(samples))

    out = {
        "mode": data.get("mode", "unknown") + "+llm_relabel",
        "num_samples": len(new_samples),
        "label_source": f"llm:{args.model}",
        "samples": new_samples,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log.info("Saved to %s", args.output)

    # Quick stats
    all_scores = [s for sample in new_samples for s in sample["label"]]
    log.info("Score stats: min=%.3f, max=%.3f, mean=%.3f",
             min(all_scores), max(all_scores),
             sum(all_scores) / len(all_scores))


if __name__ == "__main__":
    main()
