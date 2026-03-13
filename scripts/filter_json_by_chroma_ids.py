#!/usr/bin/env python3
"""
Filter a full paper JSON dump down to the exact IDs present in a ChromaDB collection.

Typical usage:
  python scripts/filter_json_by_chroma_ids.py \
    --input-json data/papers_downloaded.json \
    --output-json data/papers_downloaded_60k_for_bm25.json \
    --chromadb-path ~/chroma_db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_chroma_ids(chromadb_path: str, collection_name: str, batch_size: int) -> Set[str]:
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

    import chromadb

    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_collection(collection_name)
    total = collection.count()

    ids: Set[str] = set()
    for offset in range(0, total, batch_size):
        batch = collection.get(limit=batch_size, offset=offset, include=[])
        batch_ids = batch.get("ids", []) or []
        ids.update(str(pid) for pid in batch_ids if pid)
        print(f"Loaded Chroma IDs: {min(offset + batch_size, total)}/{total}")
    return ids


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("papers", "items", "data", "rows", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise ValueError("Unsupported JSON format: expected a list or a dict containing papers/items/data.")


def _iter_id_candidates(record: Dict[str, Any]) -> Iterable[str]:
    for key in ("paper_id", "id", "arxiv_id"):
        value = record.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                yield text


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter paper JSON to exactly the IDs present in ChromaDB")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Full paper JSON file used to build or mirror the Chroma corpus",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output subset JSON file for BM25 initialization",
    )
    parser.add_argument(
        "--chromadb-path",
        type=str,
        default=os.environ.get("CHROMA_PERSIST_DIR", os.path.expanduser("~/chroma_db")),
        help="Path to ChromaDB persist directory",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="papers",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size when reading IDs from Chroma",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser()
    output_path = Path(args.output_json).expanduser()
    chroma_path = os.path.expanduser(args.chromadb_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    print(f"Reading Chroma IDs from {chroma_path} / collection={args.collection_name} ...")
    chroma_ids = _load_chroma_ids(chroma_path, args.collection_name, args.batch_size)
    print(f"Total IDs in Chroma: {len(chroma_ids)}")

    print(f"Loading JSON from {input_path} ...")
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = _extract_records(payload)
    print(f"Total records in JSON: {len(records)}")

    subset: List[Dict[str, Any]] = []
    matched_ids: Set[str] = set()
    for record in records:
        record_id = next((pid for pid in _iter_id_candidates(record) if pid in chroma_ids), "")
        if not record_id:
            continue
        subset.append(record)
        matched_ids.add(record_id)

    missing_ids = sorted(chroma_ids - matched_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Matched records written: {len(subset)}")
    print(f"Missing Chroma IDs not found in JSON: {len(missing_ids)}")
    if missing_ids:
        print("First 10 missing IDs:", missing_ids[:10])
    print(f"Saved filtered JSON to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
