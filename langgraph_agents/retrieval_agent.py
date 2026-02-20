"""Retrieval agent: query -> vector retrieval only. Extracted from agent/retrieval_agent."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from agent.models import Paper, UserProfile
from agent.embedder import create_embedder
from agent.vector_store import create_vector_store
from agent.recall_strategies import vector_recall

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Thin retrieval: index papers, then retrieve by query using vector_recall only.
    No rule-based or ItemCF; those are in RecallAgent.
    """

    def __init__(
        self,
        embedding_backend: str = "tfidf",
        embedding_dim: int = 256,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_backend: str = "numpy",
        vector_store_path: str = "./.lancedb_index",
        top_k_vector: int = 50,
    ):
        logger.info(
            "RetrievalAgent: embedding=%s dim=%s store=%s",
            embedding_backend, embedding_dim, vector_store_backend,
        )
        self.embedder = create_embedder(
            backend=embedding_backend, dim=embedding_dim, model_name=model_name,
        )
        self._vs_backend = vector_store_backend
        self._vs_path = vector_store_path
        self.store = create_vector_store(
            backend=vector_store_backend, dim=embedding_dim, db_path=vector_store_path,
        )
        self._cache: Dict[str, Paper] = {}
        self._is_fitted = False
        self.top_k_vector = top_k_vector

    def index_papers(self, papers: List[Paper]) -> int:
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
        logger.info("Indexed %s papers (total: %s)", added, self.store.size)
        return added

    def index_papers_precomputed(self, papers: List[Paper]) -> int:
        """Use papers' existing embeddings."""
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
        if not self._is_fitted:
            texts = [p.text_for_embedding() for p in valid]
            self.embedder.fit(texts)
            self._is_fitted = True
            logger.info("Embedder fitted for query encoding")
        logger.info("Indexed %s precomputed papers (total: %s, dim=%s)", added, self.store.size, emb_dim)
        return added

    def retrieve_by_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Paper]:
        """
        Query -> vector retrieval only. No user profile blending.
        """
        try:
            query_vec = self.embedder.encode(query)
        except Exception as e:
            logger.warning("Encode failed: %s", e)
            return []
        if self.store.size == 0:
            return []
        results = vector_recall(query_vec, self.store, top_k=top_k * 3)
        papers = []
        for pid, score in results[:top_k]:
            paper = self._cache.get(pid)
            if paper:
                papers.append(Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories, published=paper.published,
                    score=score,
                ))
        logger.info("retrieve_by_query: %s papers (top_k=%s)", len(papers), top_k)
        return papers

    def save_index(self, path: str | Path) -> None:
        self.store.save(path)

    def load_index(self, path: str | Path) -> None:
        self.store.load(path)

    @property
    def index_size(self) -> int:
        return self.store.size
