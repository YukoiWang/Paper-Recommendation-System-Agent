"""Retrieval agent: query -> HyDE (optional) + vector + BM25 hybrid, RRF fusion. QA path: no history filter."""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent.models import Paper, UserProfile
from agent.embedder import create_embedder
from agent.vector_store import create_vector_store, ChromaDBVectorStore
from agent.recall_strategies import vector_recall, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore


class RetrievalAgent:
    """
    Query-driven retrieval: optional HyDE (caller provides), vector + BM25 hybrid, RRF fusion.
    QA path: do not filter by read/block.
    """

    def __init__(
        self,
        embedding_backend: str = "tfidf",
        embedding_dim: int = 256,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_backend: str = "numpy",
        vector_store_path: str = "./.lancedb_index",
        top_k_vector: int = 50,
        chromadb_path: str = None,
        collection_name: str = "papers",
        use_bm25: bool = True,
        rrf_k: int = 60,
    ):
        if chromadb_path is None:
            chromadb_path = os.path.expanduser("~/chroma_db")
        self._vs_backend = vector_store_backend
        self._vs_path = vector_store_path
        self._use_chromadb = (vector_store_backend == "chromadb")
        self._use_bm25 = use_bm25 and (BM25Okapi is not None) and not self._use_chromadb
        self._rrf_k = rrf_k
        self._bm25_index: Any = None
        self._bm25_ids: List[str] = []

        if self._use_chromadb:
            logger.info("RetrievalAgent: using ChromaDB at %s", chromadb_path)
            self.store = create_vector_store(
                backend="chromadb",
                chromadb_path=chromadb_path,
                collection_name=collection_name,
            )
            self.embedder = create_embedder(
                backend="sentence_transformer",
                model_name=model_name,
            )
            self._is_fitted = True
        else:
            logger.info(
                "RetrievalAgent: embedding=%s dim=%s store=%s",
                embedding_backend, embedding_dim, vector_store_backend,
            )
            self.embedder = create_embedder(
                backend=embedding_backend, dim=embedding_dim, model_name=model_name,
            )
            self.store = create_vector_store(
                backend=vector_store_backend, dim=embedding_dim, db_path=vector_store_path,
            )
            self._is_fitted = False

        self._cache: Dict[str, Paper] = {}
        self.top_k_vector = top_k_vector

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _build_bm25(self, ids: List[str], texts: List[str]) -> None:
        if not self._use_bm25 or BM25Okapi is None or not ids:
            return
        corpus = [self._tokenize(t) for t in texts]
        self._bm25_index = BM25Okapi(corpus)
        self._bm25_ids = ids
        logger.info("BM25 index built: %s docs", len(ids))

    def index_papers(self, papers: List[Paper]) -> int:
        if self._use_chromadb:
            logger.info("ChromaDB mode: skipping index_papers (data already loaded)")
            return 0
        if not papers:
            return 0
        texts = [p.text_for_embedding() for p in papers]
        if not self._is_fitted:
            self.embedder.fit(texts)
            self._is_fitted = True
            self.store = create_vector_store(
                backend=self._vs_backend, dim=self.embedder.dim, db_path=self._vs_path,
            )
            logger.info("Embedder fitted, dim=%s", self.embedder.dim)
        vectors = self.embedder.encode_batch(texts)
        ids = [p.paper_id for p in papers]
        added = self.store.add(ids, vectors)
        for p, vec in zip(papers, vectors):
            p.embedding = vec
            self._cache[p.paper_id] = p
        self._build_bm25(ids, texts)
        logger.info("Indexed %s papers (total: %s)", added, self.store.size)
        return added

    def index_papers_precomputed(self, papers: List[Paper]) -> int:
        if self._use_chromadb:
            logger.info("ChromaDB mode: skipping index_papers_precomputed")
            return 0
        valid = [p for p in papers if p.embedding is not None]
        if not valid:
            logger.warning("No precomputed embeddings; falling back to index_papers.")
            return self.index_papers(papers)
        emb_dim = valid[0].embedding.shape[0]
        if self.embedder.dim != emb_dim:
            logger.warning(
                "Embedder dim %s != precomputed %s; re-embedding.",
                self.embedder.dim, emb_dim,
            )
            return self.index_papers(papers)
        self.store = create_vector_store(
            backend=self._vs_backend, dim=emb_dim, db_path=self._vs_path,
        )
        vectors = np.vstack([p.embedding for p in valid])
        ids = [p.paper_id for p in valid]
        added = self.store.add(ids, vectors)
        for p in valid:
            self._cache[p.paper_id] = p
        texts = [p.text_for_embedding() for p in valid]
        self._build_bm25(ids, texts)
        if not self._is_fitted:
            self.embedder.fit(texts)
            self._is_fitted = True
            logger.info("Embedder fitted for query encoding")
        logger.info("Indexed %s precomputed papers (total: %s, dim=%s)", added, self.store.size, emb_dim)
        return added

    def retrieve_by_query(
        self,
        query: str,
        top_k: int = 10,
        hyde_document: Optional[str] = None,
    ) -> List[Paper]:
        """
        QA path: use hyde_document for vector encoding if provided (short-query HyDE from planner);
        BM25 always uses the original query. RRF fuses dense + sparse when both available.
        """
        vector_query = (hyde_document or "").strip() or query
        try:
            query_vec = self.embedder.encode(vector_query)
        except Exception as e:
            logger.warning("Encode failed: %s", e)
            return []
        if self.store.size == 0:
            return []

        if self._use_chromadb and isinstance(self.store, ChromaDBVectorStore):
            hits = self.store.search_with_metadata(query_vec, top_k=top_k)
            papers = []
            for h in hits:
                papers.append(Paper(
                    paper_id=h["paper_id"],
                    title=h.get("title", ""),
                    abstract=h.get("abstract", ""),
                    authors=[],
                    categories=[],
                    published=str(h.get("year", "")),
                    score=h["score"],
                ))
            logger.info("retrieve_by_query (ChromaDB): %s papers (top_k=%s)", len(papers), top_k)
            return papers

        fetch_k = max(top_k * 2, 40)
        dense_results = vector_recall(query_vec, self.store, top_k=fetch_k)

        if self._bm25_index and self._bm25_ids and query.strip():
            q_tokens = self._tokenize(query)
            if q_tokens:
                bm25_scores = self._bm25_index.get_scores(q_tokens)
                sparse_results: List[Tuple[str, float]] = [
                    (self._bm25_ids[i], float(bm25_scores[i]))
                    for i in range(len(self._bm25_ids))
                    if bm25_scores[i] > 0
                ]
                sparse_results.sort(key=lambda x: x[1], reverse=True)
                sparse_results = sparse_results[:fetch_k]
                fused = reciprocal_rank_fusion(
                    dense_results, sparse_results, k=self._rrf_k, top_n=top_k
                )
            else:
                fused = dense_results[:top_k]
        else:
            fused = dense_results[:top_k]

        papers = []
        for pid, score in fused:
            paper = self._cache.get(pid)
            if paper:
                papers.append(Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories, published=paper.published,
                    score=score,
                ))
        logger.info("retrieve_by_query: %s papers (top_k=%s, hyde=%s)", len(papers), top_k, bool(hyde_document))
        return papers

    def save_index(self, path: str | Path) -> None:
        self.store.save(path)

    def load_index(self, path: str | Path) -> None:
        self.store.load(path)

    @property
    def index_size(self) -> int:
        return self.store.size
