"""Load papers: built-in corpus, or Lance dataset (local/HF) with optional prefer_recent."""
from __future__ import annotations
import logging
import os
from typing import List, Optional
import numpy as np

from .models import Paper

logger = logging.getLogger(__name__)

ML_CATEGORIES = [
    "cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.IR",
    "cs.NE", "stat.ML", "cs.RO", "cs.MA",
]

_BUILTIN = [
    ("2401.00001", "Efficient Sparse Attention for Long-Context Language Models",
     "Sparse attention that reduces transformer complexity to near-linear for 128k+ tokens.",
     ["Yichen Wang", "Sarah Chen", "Michael Roberts"], ["cs.CL", "cs.LG"], "2025-01-15"),
    ("2401.00002", "Scaling Laws for Mixture-of-Experts Language Models",
     "Scaling behavior of MoE architectures and new scaling laws for expert specialization.",
     ["Alex Thompson", "Wei Liu", "Priya Patel"], ["cs.LG", "cs.CL"], "2025-01-18"),
    ("2401.00003", "Constitutional AI: Training with Human Feedback and Self-Critique",
     "Aligning LLMs with human values via RLHF and model self-critique.",
     ["David Park", "Emma Wilson", "Raj Gupta"], ["cs.AI", "cs.CL"], "2025-01-20"),
    ("2401.00004", "LoRA-XS: Extremely Parameter-Efficient Fine-Tuning with Rank-1 Adaptation",
     "Rank-1 low-rank adaptation; 0.01% params, 95% of full fine-tuning performance.",
     ["Sophia Lee", "James Brown"], ["cs.CL", "cs.LG"], "2025-01-22"),
    ("2401.00005", "Retrieval-Augmented Generation with Dynamic Knowledge Graphs",
     "RAG over a dynamic knowledge graph instead of a flat document store.",
     ["Min Zhang", "Carlos Garcia", "Anna Kowalski"], ["cs.CL", "cs.IR"], "2025-01-25"),
    ("2401.00010", "Vision Transformers with Rotary Position Embeddings",
     "Rotary position embeddings adapted to vision transformers for varying resolutions.",
     ["Lisa Wang", "Tom Anderson"], ["cs.CV", "cs.LG"], "2025-01-12"),
    ("2401.00011", "Self-Supervised Visual Learning via Masked Image Modeling at Scale",
     "Masked image modeling scaled to 10B vision models on 2B images.",
     ["Kevin Zhou", "Maria Santos", "Hiroshi Yamada"], ["cs.CV", "cs.LG"], "2025-01-14"),
    ("2401.00020", "Offline RL with Conservative World Models",
     "Conservative world model from offline data underestimating OOD action values.",
     ["Chris Taylor", "Mei Lin"], ["cs.LG", "cs.AI"], "2025-01-10"),
    ("2401.00030", "Scalable Graph Transformers for Molecular Property Prediction",
     "Graph transformer scaling to molecular graphs with thousands of atoms.",
     ["Sara Hooker", "David Chen", "Isabella Martinez"], ["cs.LG"], "2025-01-11"),
    ("2401.00040", "Consistency Models: Fast One-Step Generation",
     "Consistency models mapping noise to data in a single step.",
     ["Alex Kim", "Laura White", "Raj Mehta"], ["cs.LG", "cs.CV"], "2025-01-09"),
    ("2401.00050", "Communication-Efficient Federated Learning with Gradient Compression",
     "Adaptive gradient compression for federated learning, 100x less communication.",
     ["Monica Patel", "Hans Schmidt"], ["cs.LG", "cs.DC"], "2025-01-12"),
    ("2401.00100", "Chain-of-Thought Reasoning from Pretraining Data Composition",
     "How pretraining data composition affects CoT emergence in language models.",
     ["Kevin Zhou", "Anna Kowalski"], ["cs.CL", "cs.AI"], "2025-01-10"),
]


def load_builtin_corpus() -> List[Paper]:
    papers = [
        Paper(paper_id=pid, title=t, abstract=a, authors=au, categories=c, published=pub)
        for pid, t, a, au, c, pub in _BUILTIN
    ]
    logger.info("Loaded %s built-in papers", len(papers))
    return papers


DEFAULT_LANCE_PATH = "./data/lance_dataset"


def _load_lance_recent(ds, total_rows: int, max_papers: int, filter_set: Optional[set] = None):
    """Two-phase load: sort by date desc, filter by categories, then take full rows."""
    import pandas as pd
    import pyarrow.compute as pc

    logger.info("Phase 1: date+categories for %s rows", total_rows)
    light_table = ds.to_table(columns=["update_date", "categories"])
    sort_indices = pc.sort_indices(light_table, sort_keys=[("update_date", "descending")])
    idx_list = sort_indices.to_pylist()
    sorted_table = light_table.take(sort_indices)
    df_light = sorted_table.to_pandas()
    df_light["_orig_idx"] = idx_list

    if filter_set:
        def _match(cats_raw):
            if isinstance(cats_raw, str):
                cats = cats_raw.replace(",", " ").split()
            elif isinstance(cats_raw, list):
                cats = cats_raw
            else:
                return False
            return any(c.lower() in filter_set for c in cats)

        mask = df_light["categories"].apply(_match)
        df_light = df_light[mask]
        logger.info("After category filter: %s rows", len(df_light))

    df_light = df_light.head(max_papers)
    target_indices = df_light["_orig_idx"].tolist()
    logger.info("Phase 1 done: %s papers", len(target_indices))

    if not target_indices:
        return pd.DataFrame()

    logger.info("Phase 2: full data for %s papers", len(target_indices))
    return ds.take(target_indices).to_pandas()


def load_from_lance_hf(
    dataset_name: str = "davanstrien/arxiv-cs-papers-lance",
    lance_path: Optional[str] = None,
    max_papers: Optional[int] = None,
    filter_categories: Optional[List[str]] = None,
    prefer_recent: bool = True,
) -> List[Paper]:
    """
    Load from local Lance dir or HuggingFace. If prefer_recent and max_papers set,
    load newest papers first (by update_date). Requires pylance (local) or lancedb (remote).
    """
    import pandas as pd

    local_path = lance_path or DEFAULT_LANCE_PATH
    filter_set = set(c.lower() for c in filter_categories) if filter_categories else None

    if local_path and os.path.isdir(local_path):
        try:
            import lance as _lance
        except ImportError as e:
            raise ImportError("Local Lance requires: pip install pylance") from e
        logger.info("Loading Lance from %s", local_path)
        ds = _lance.dataset(local_path)
        total_rows = ds.count_rows()
        logger.info("Lance rows: %s", total_rows)
        if prefer_recent and max_papers and max_papers > 0:
            df = _load_lance_recent(ds, total_rows, max_papers, filter_set)
        else:
            fetch_n = (max_papers * 3) if (max_papers and max_papers > 0) else total_rows
            fetch_n = min(fetch_n, total_rows)
            df = ds.head(fetch_n).to_pandas()
    else:
        # HuggingFace: 优先用 datasets 流式加载（避免 pylance 读 lance 时的 panic）
        import tempfile
        _cache = os.path.join(tempfile.gettempdir(), "hf_datasets_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", _cache)
        df = None
        try:
            from datasets import load_dataset
            logger.info("Loading davanstrien/arxiv-example from HuggingFace (streaming)...")
            stream = load_dataset(
                "davanstrien/arxiv-example",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            need = (max_papers * 2) if (max_papers and max_papers > 0) else None  # None = no limit
            rows = []
            for i, row in enumerate(stream):
                if need is not None and i >= need:
                    break
                rows.append(dict(row))
            df = pd.DataFrame(rows)
        except Exception as e:
            logger.warning("datasets streaming failed: %s", e)
            df = None
        if df is None:
            raise RuntimeError(
                "Could not load arxiv papers. Try: pip install datasets, "
                "set HF_DATASETS_CACHE=/tmp/hf_cache (for disk space), "
                "or use lance_path= to a local lance directory."
            )

    logger.info("Loaded %s rows", len(df))

    vector_col = None
    for c in ("vector", "embedding", "embeddings", "vec"):
        if c in df.columns:
            vector_col = c
            break
    if not vector_col:
        logger.warning("No embedding column found")

    papers = []
    for _, row in df.iterrows():
        if max_papers and max_papers > 0 and len(papers) >= max_papers:
            break
        title = str(row.get("title", ""))
        if not title:
            continue
        abstract = str(row.get("abstract", ""))
        paper_id = str(row.get("id", row.get("arxiv_id", "")))
        cats_raw = row.get("categories", row.get("subjects", row.get("primary_subject", "")))
        if isinstance(cats_raw, str):
            categories = cats_raw.replace(",", " ").split()
        elif isinstance(cats_raw, list):
            categories = cats_raw
        else:
            categories = []
        if filter_set and not any(c.lower() in filter_set for c in categories):
            continue
        authors_raw = row.get("authors", "")
        if isinstance(authors_raw, list):
            authors = authors_raw
        elif isinstance(authors_raw, str):
            authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        else:
            authors = []
        published = row.get(
            "update_date",
            row.get("published", row.get("submission_date", row.get("publication_date", ""))),
        )
        if hasattr(published, "strftime"):
            published = published.strftime("%Y-%m-%d")
        else:
            published = str(published) if published else ""
        embedding = None
        if vector_col and row.get(vector_col) is not None:
            vec = row[vector_col]
            if isinstance(vec, (list, np.ndarray)):
                embedding = np.asarray(vec, dtype=np.float32)
        papers.append(Paper(
            paper_id=paper_id,
            title=title.strip().replace("\n", " "),
            abstract=abstract.strip().replace("\n", " ") if abstract else "",
            authors=authors[:10],
            categories=categories,
            published=published,
            embedding=embedding,
        ))

    n_emb = sum(1 for p in papers if p.embedding is not None)
    dim = papers[0].embedding.shape[0] if n_emb else 0
    logger.info("Converted %s papers (%s with %s-dim embeddings)", len(papers), n_emb, dim)
    return papers
