"""
Load Papers into Database and Vector DB.

Sources:
  - json: 从 JSON 文件加载
  - lance: 从 Lance/HuggingFace 数据集加载 (默认 davanstrien/arxiv-cs-papers-lance)

Run from project root: python scripts/load_papers.py [lance|json] [path_or_dataset]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from typing import Any, Dict, List, Optional

from loguru import logger

# 用 pysqlite3 替代系统 sqlite3，满足 ChromaDB 对 sqlite >= 3.35.0 的要求
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

# Stub posthog before ChromaDB loads (Python 3.8 compatibility)
class _PosthogStub:
    disabled = True
    project_api_key = ""

    @staticmethod
    def capture(*args, **kwargs):
        pass


sys.modules["posthog"] = _PosthogStub()

# 可选：metadata_db 需要 sqlalchemy/PostgreSQL，chroma-only 时可不装
try:
    from backend.services.metadata_db import MetadataDBService
except ImportError:
    MetadataDBService = None  # type: ignore

from backend.services.vector_db import VectorDBService
from backend.services.embedding import EmbeddingService


def _paper_to_dict(p: Any) -> Dict[str, Any]:
    """Convert agent Paper or dict to backend format."""
    if isinstance(p, dict):
        base = {
            "paper_id": str(p.get("paper_id", p.get("id", ""))),
            "title": str(p.get("title", "") or ""),
            "abstract": str(p.get("abstract", "") or ""),
            "authors": p.get("authors", []) or [],
            "venue": str(p.get("venue", "") or ""),
            "year": p.get("year"),
        }
        # year: metadata_db expects int or None
        if base["year"] is not None and base["year"] != "":
            try:
                base["year"] = int(str(base["year"])[:4])
            except (ValueError, TypeError):
                base["year"] = None
        else:
            base["year"] = None
        return base
    return {
        "paper_id": getattr(p, "paper_id", str(getattr(p, "id", ""))),
        "title": getattr(p, "title", ""),
        "abstract": getattr(p, "abstract", ""),
        "authors": getattr(p, "authors", []) or [],
        "venue": "",
        "year": (getattr(p, "published", "") or "")[:4] if getattr(p, "published", None) else "",
    }


async def load_papers_from_file(file_path: str) -> None:
    """
    从JSON文件加载论文到 metadata_db（PostgreSQL）和 vector_db（Chroma）。
    需 PostgreSQL 运行且配置正确，否则 metadata_db 写入会跳过。
    """
    metadata_db = None
    if MetadataDBService:
        try:
            metadata_db = MetadataDBService()
            logger.info("Metadata DB (PostgreSQL) enabled, will write papers")
        except Exception as e:
            logger.warning(f"Metadata DB init failed (skip meta_db write): {e}")
    vector_db = VectorDBService()
    embedding_service = EmbeddingService()

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    papers = raw if isinstance(raw, list) else [raw]
    papers = [_paper_to_dict(p) for p in papers]

    logger.info(f"Loading {len(papers)} papers from JSON...")
    batch_size = 100
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        texts = [f"{p.get('title','')} {p.get('abstract','')}".strip() for p in batch]
        embeddings = await embedding_service.embed_texts(texts, batch_size=batch_size)
        if metadata_db:
            try:
                await metadata_db.upsert_papers(batch)
            except Exception as e:
                logger.warning(f"Metadata DB skip: {e}")
        await vector_db.add_papers(batch, embeddings)
        logger.info(f"Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")
    logger.info("All papers loaded successfully!")


async def load_papers_from_lance(
    dataset_name: str = "davanstrien/arxiv-cs-papers-lance",
    max_papers: Optional[int] = None,
    filter_categories: Optional[List[str]] = None,
    chroma_only: bool = False,
) -> None:
    """
    从 Lance/HuggingFace 数据集加载论文到向量库。
    使用数据集自带的 embedding（如有），否则用本地 embedding 模型。
    """
    try:
        from agent.data_loader import load_from_lance_hf
    except ImportError:
        raise ImportError("Lance 数据源需要: pip install lancedb")

    logger.info(f"Loading from {dataset_name} (max={max_papers})...")
    papers = load_from_lance_hf(
        dataset_name=dataset_name,
        max_papers=max_papers,
        filter_categories=filter_categories,
        prefer_recent=True,
    )
    if not papers:
        logger.warning("No papers loaded from Lance dataset")
        return

    vector_db = VectorDBService()
    embedding_service = EmbeddingService()
    metadata_db = None
    if not chroma_only and MetadataDBService:
        try:
            metadata_db = MetadataDBService()
        except Exception as e:
            logger.warning(f"Metadata DB unavailable (chroma-only mode): {e}")

    batch_size = 100
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        dict_batch = [_paper_to_dict(p) for p in batch]

        # 使用数据集自带的 embedding（如有），否则用本地模型
        embeddings: List[Optional[List[float]]] = [None] * len(batch)
        need_embed_idxs: List[int] = []
        for j, p in enumerate(batch):
            emb = getattr(p, "embedding", None)
            if emb is not None:
                try:
                    arr = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    if arr and isinstance(arr[0], (int, float)):
                        embeddings[j] = arr
                except Exception:
                    pass
            if embeddings[j] is None:
                need_embed_idxs.append(j)

        if need_embed_idxs:
            need_texts = [
                f"{dict_batch[j].get('title','')} {dict_batch[j].get('abstract','')}".strip()
                for j in need_embed_idxs
            ]
            sub_embs = await embedding_service.embed_texts(need_texts, batch_size=len(need_texts))
            for idx, emb in zip(need_embed_idxs, sub_embs):
                embeddings[idx] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        if any(e is None for e in embeddings):
            texts = [f"{p.get('title','')} {p.get('abstract','')}".strip() for p in dict_batch]
            embeddings = await embedding_service.embed_texts(texts, batch_size=batch_size)
        emb_final: List[List[float]] = [e if isinstance(e, list) else e.tolist() for e in embeddings]

        if metadata_db:
            try:
                await metadata_db.upsert_papers(dict_batch)
            except Exception as e:
                logger.warning(f"Metadata DB skip: {e}")

        await vector_db.add_papers(dict_batch, emb_final)
        logger.info(f"Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")

    logger.info("Lance papers loaded successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load papers into vector DB (and optionally metadata DB)"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="lance",
        choices=["json", "lance"],
        help="Data source: json file or lance dataset (default: lance)",
    )
    parser.add_argument(
        "path_or_name",
        nargs="?",
        default="davanstrien/arxiv-cs-papers-lance",
        help="For json: file path. For lance: dataset name (default: davanstrien/arxiv-cs-papers-lance)",
    )
    parser.add_argument(
        "-n",
        "--max-papers",
        type=int,
        default=None,
        help="Max papers to load (lance only, default: no limit)",
    )
    parser.add_argument(
        "--chroma-only",
        action="store_true",
        help="Skip metadata DB, only write to Chroma",
    )
    parser.add_argument(
        "--filter-categories",
        type=str,
        default=None,
        help="Comma-separated categories to filter (lance only, e.g. cs.LG,cs.AI)",
    )
    args = parser.parse_args()

    if args.source == "json":
        if not args.path_or_name or not Path(args.path_or_name).exists():
            parser.error("JSON source requires an existing file path")
        asyncio.run(load_papers_from_file(args.path_or_name))
    else:
        cats = [c.strip() for c in args.filter_categories.split(",")] if args.filter_categories else None
        asyncio.run(
            load_papers_from_lance(
                dataset_name=args.path_or_name,
                max_papers=args.max_papers,
                filter_categories=cats,
                chroma_only=args.chroma_only,
            )
        )


if __name__ == "__main__":
    main()
