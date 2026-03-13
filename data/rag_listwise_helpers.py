# -*- coding: utf-8 -*-
"""
Helpers for listwise data generation: RAG vector DB, metadata DB, paper API.
Uses backend services when run from project root; can fallback to Chroma-only.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# List paper IDs from Chroma (RAG vector DB)
# ---------------------------------------------------------------------------
def get_seed_paper_ids_from_rag(limit: int = 300, chroma_path: Optional[str] = None) -> List[str]:
    """List paper IDs from RAG Chroma collection (for sampling seed papers)."""
    path = chroma_path or os.getenv("CHROMA_PERSIST_DIR") or os.path.expanduser("~/chroma_db")
    path = Path(path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        coll = client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})
        # Chroma get with limit returns first `limit` items
        res = coll.get(limit=limit, include=[])
        ids = list(res.get("ids", []))
        return ids
    except Exception as e:
        print(f"get_seed_paper_ids_from_rag error: {e}")
        return []


# ---------------------------------------------------------------------------
# Backend services (async) - VectorDB, MetadataDB, Embedding
# ---------------------------------------------------------------------------
_CACHED_VEC_DB = None
_CACHED_META_DB = None
_CACHED_EMB_SVC = None


def _get_vec_db():
    """Get cached VectorDBService (Chroma/Milvus)."""
    global _CACHED_VEC_DB
    if _CACHED_VEC_DB is not None:
        return _CACHED_VEC_DB
    try:
        from backend.services.vector_db import VectorDBService

        _CACHED_VEC_DB = VectorDBService()
        return _CACHED_VEC_DB
    except Exception as e:
        print(f"VectorDBService init error: {e}")
        return None


def _get_meta_db():
    """Get cached MetadataDBService (PostgreSQL). If unavailable, returns None."""
    global _CACHED_META_DB
    if _CACHED_META_DB is not None:
        return _CACHED_META_DB
    try:
        from backend.services.metadata_db import MetadataDBService

        _CACHED_META_DB = MetadataDBService()
        return _CACHED_META_DB
    except Exception:
        # Metadata DB is optional for offline eval; avoid noisy logs.
        return None


def _get_emb_svc():
    """Get cached EmbeddingService. Only needed when adding new papers to RAG."""
    global _CACHED_EMB_SVC
    if _CACHED_EMB_SVC is not None:
        return _CACHED_EMB_SVC
    try:
        from backend.services.embedding import EmbeddingService

        _CACHED_EMB_SVC = EmbeddingService()
        return _CACHED_EMB_SVC
    except Exception as e:
        print(f"EmbeddingService init error: {e}")
        return None


def get_paper_from_rag(paper_id: str) -> Optional[Dict[str, Any]]:
    """Get paper by id from metadata DB (title, abstract, etc.). Returns None if not in RAG."""
    meta_db = _get_meta_db()
    if meta_db is None:
        return None
    return _run(meta_db.get_paper_by_id(paper_id))


def get_papers_from_rag(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Batch get papers from metadata DB."""
    meta_db = _get_meta_db()
    if meta_db is None or not paper_ids:
        return []
    return _run(meta_db.get_papers_by_ids(paper_ids))


def get_embedding_for_paper(paper_id: str) -> Optional[List[float]]:
    """Get embedding vector for paper_id from vector DB."""
    vec_db = _get_vec_db()
    if vec_db is None:
        return None
    return _run(vec_db.get_embedding(paper_id))


def similarity_search_rag(
    query_embedding: List[float],
    top_k: int = 500,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search RAG vector DB by query embedding. Returns list of {paper_id, score, metadata}."""
    vec_db = _get_vec_db()
    if vec_db is None:
        return []
    return _run(vec_db.similarity_search(query_embedding, top_k=top_k, exclude_ids=exclude_ids or []))


def add_paper_to_rag(paper: Dict[str, Any]) -> bool:
    """Insert paper into metadata DB and vector DB (compute embedding and add)."""
    vec_db = _get_vec_db()
    meta_db = _get_meta_db()
    emb_svc = _get_emb_svc()
    if meta_db is None or vec_db is None or emb_svc is None:
        return False
    paper_id = paper.get("paper_id") or paper.get("id", "")
    if not paper_id:
        return False

    async def _add():
        await meta_db.upsert_paper({
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
        })
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
        emb = await emb_svc.embed_query(text)
        await vec_db.add_papers([paper], [emb])
        return True

    try:
        return _run(_add())
    except Exception as e:
        print(f"add_paper_to_rag error: {e}")
        return False


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity (assume vectors are normalized)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
