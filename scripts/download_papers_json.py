#!/usr/bin/env python3
"""
Download papers from HuggingFace davanstrien/arxiv-cs-papers-lance and save as JSON.

Uses the dataset's Parquet conversion (no trust_remote_code). Then run:
  CHROMA_PERSIST_DIR=/tmp/chroma_db python scripts/load_papers.py json data/papers_downloaded.json

Usage:
  python scripts/download_papers_json.py -n 5000 -o data/papers_downloaded.json
  python scripts/download_papers_json.py -n 10000 --cache-dir /tmp/hf_cache
"""
import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATASET_NAME = "davanstrien/arxiv-cs-papers-lance"
PARQUET_REVISION = "refs/convert/parquet"


def _row_to_paper(row: dict) -> dict:
    """Map dataset row to load_papers JSON format (paper_id, title, abstract, authors, venue, year)."""
    paper_id = str(row.get("id", row.get("arxiv_id", row.get("paper_id", ""))))
    title = str(row.get("title", "") or "").replace("\n", " ").strip()
    abstract = str(row.get("abstract", "") or "").replace("\n", " ").strip()
    authors_raw = row.get("authors", row.get("author", []))
    if isinstance(authors_raw, list):
        authors = [str(a) for a in authors_raw[:20]]
    elif isinstance(authors_raw, str):
        authors = [a.strip() for a in authors_raw.split(",") if a.strip()][:20]
    else:
        authors = []
    venue = str(row.get("primary_subject", row.get("categories", row.get("venue", ""))) or "")
    if isinstance(venue, list):
        venue = venue[0] if venue else ""
    pub = row.get("update_date", row.get("submission_date", row.get("published", row.get("publication_date", ""))))
    if hasattr(pub, "year"):
        year = int(pub.year)
    elif isinstance(pub, str) and len(pub) >= 4:
        try:
            year = int(pub[:4])
        except ValueError:
            year = None
    else:
        year = None
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "venue": venue,
        "year": year,
    }


def main():
    parser = argparse.ArgumentParser(description="Download davanstrien/arxiv-cs-papers-lance to JSON")
    parser.add_argument("-n", "--max-papers", type=int, default=5000,
                        help="Max papers to take (default 5000)")
    parser.add_argument("-o", "--output", type=str,
                        default=str(_PROJECT_ROOT / "data" / "papers_downloaded.json"),
                        help="Output JSON path")
    parser.add_argument("--cache-dir", type=str, default="",
                        help="HF_DATASETS_CACHE dir (default: /tmp/hf_cache)")
    args = parser.parse_args()

    cache = args.cache_dir or os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache
    Path(cache).mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        sys.exit(1)

    print(f"Loading {DATASET_NAME} (parquet revision, cache={cache})...")
    try:
        ds = load_dataset(
            DATASET_NAME,
            split="train",
            revision=PARQUET_REVISION,
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"load_dataset error: {e}")
        sys.exit(1)

    n_total = len(ds)
    take = min(args.max_papers, n_total)
    print(f"Dataset rows: {n_total}, taking first {take}")

    out = []
    for i in range(take):
        row = ds[i]
        if isinstance(row, dict):
            rec = _row_to_paper(row)
        else:
            rec = _row_to_paper(dict(zip(ds.column_names, row)))
        if not rec["paper_id"] or not rec["title"]:
            continue
        out.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{take}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} papers to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
