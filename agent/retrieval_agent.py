"""Retrieval agent: index papers, multi-path recall, merge, return recommendations."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from agent.models import Paper, RecommendationResult, UserProfile
from embedder import create_embedder
from vector_store import create_vector_store
from cold_start import resolve_user_vector, get_trending_papers
from recall_strategies import vector_recall, rule_based_recall, itemcf_recall, merge_results

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Index papers (embed or use precomputed vectors), then retrieve for user via
    vector + rule + ItemCF recall, merge by score sum, optional trending supplement.
    """

    def __init__(
        self,
        embedding_backend: str = "tfidf",
        embedding_dim: int = 256,
        model_name: str = "all-MiniLM-L6-v2",
        vector_store_backend: str = "numpy",
        vector_store_path: str = "./.lancedb_index",
        top_k_vector: int = 50,
        top_k_itemcf: int = 20,
        top_k_final: int = 30,
        author_boost: float = 0.25,
        category_boost: float = 0.10,
        trending_count: int = 10,
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
        self.top_k_itemcf = top_k_itemcf
        self.top_k_final = top_k_final
        self.author_boost = author_boost
        self.category_boost = category_boost
        self.trending_count = trending_count

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
        """Use papers' existing embeddings. Embedder dim must match precomputed dim."""
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

    def retrieve_for_user(
        self,
        user: UserProfile,
        new_papers: Optional[List[Paper]] = None,
        top_k_override: Optional[int] = None,
        prefer_recent: bool = False,
    ) -> RecommendationResult:
        """Retrieve for user profile. top_k_override from blackboard; prefer_recent for latest papers."""
        all_papers = list(new_papers) if new_papers else []
        if not all_papers:
            all_papers = list(self._cache.values())
        if new_papers:
            self.index_papers(new_papers)
        if self.store.size == 0:
            logger.warning("Vector store empty")
            return RecommendationResult(user_id=user.user_id, recall_stats={"error": "empty"})
        top_k = top_k_override if top_k_override is not None else self.top_k_final
        user_vec, is_cold = resolve_user_vector(user, self.embedder, all_papers)
        stats = {}
        vec_results = vector_recall(user_vec, self.store, top_k=self.top_k_vector)
        stats["vector_recall"] = len(vec_results)
        rule_results = rule_based_recall(
            user, all_papers,
            author_boost=self.author_boost,
            category_boost=self.category_boost,
        )
        stats["rule_recall"] = len(rule_results)
        icf_results = itemcf_recall(user, self.store, self._cache, top_k_total=self.top_k_itemcf)
        stats["itemcf_recall"] = len(icf_results)
        merged = merge_results(vec_results, rule_results, icf_results, top_k=top_k)
        stats["merged"] = len(merged)
        if is_cold and len(merged) < self.trending_count:
            existing_ids = {pid for pid, _ in merged}
            for tp in get_trending_papers(all_papers, self.trending_count):
                if tp.paper_id not in existing_ids:
                    merged.append((tp.paper_id, 0.0))
                    existing_ids.add(tp.paper_id)
                if len(merged) >= top_k:
                    break
            stats["trending_supplement"] = len(merged) - stats["merged"]
        recommended = []
        for pid, score in merged:
            paper = self._cache.get(pid)
            if paper:
                rec = Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories,
                    published=paper.published, embedding=paper.embedding, score=score,
                )
                recommended.append(rec)
        if prefer_recent and recommended:
            recommended = self._apply_recency_boost(recommended)
            recommended = recommended[:top_k]
        result = RecommendationResult(user_id=user.user_id, recommended_papers=recommended, recall_stats=stats)
        logger.info("Stats: %s | final %s papers (top_k=%s, prefer_recent=%s)", stats, len(recommended), top_k, prefer_recent)
        return result

    def _apply_recency_boost(self, papers: List[Paper], recency_weight: float = 0.3) -> List[Paper]:
        """Sort by score * (1 - recency_weight) + recency_score * recency_weight."""
        import re
        def parse_year(pub: str) -> int:
            if not pub:
                return 0
            m = re.search(r"20\d{2}", str(pub))
            return int(m.group()) if m else 0
        max_year = max(parse_year(p.published) for p in papers) or 2024
        scored = []
        for p in papers:
            yr = parse_year(p.published)
            recency = max(0, (yr - 2020) / 4.0) if max_year > 2020 else 0
            norm_recency = recency / 4.0 if max_year > 2020 else 0
            combined = p.score * (1 - recency_weight) + norm_recency * recency_weight
            scored.append((combined, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def retrieve_by_query(
        self,
        query: str,
        user: UserProfile,
        top_k: int = 10,
        prefer_recent: bool = False,
    ) -> List[Paper]:
        """
        Query-based retrieval (for user active question). Combines query vector recall
        with user profile recall. Reads top_k from blackboard.
        """
        from recall_strategies import vector_recall
        candidates: Dict[str, float] = {}
        try:
            query_vec = self.embedder.encode(query)
        except Exception as e:
            logger.warning("Encode failed: %s", e)
            return []
        if self.store.size == 0:
            return []
        for pid, score in vector_recall(query_vec, self.store, top_k=top_k * 3):
            candidates[pid] = candidates.get(pid, 0.0) + score
        if user.interest_text or user.interest_vector is not None:
            user_vec, _ = resolve_user_vector(user, self.embedder, list(self._cache.values()))
            for pid, score in vector_recall(user_vec, self.store, top_k=top_k):
                candidates[pid] = candidates.get(pid, 0.0) + score * 0.3
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        papers = []
        for pid, score in sorted_candidates[:top_k]:
            paper = self._cache.get(pid)
            if paper:
                papers.append(Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories, published=paper.published,
                    score=score,
                ))
        if prefer_recent and papers:
            papers = self._apply_recency_boost(papers)[:top_k]
        logger.info("retrieve_by_query: %s papers (top_k=%s)", len(papers), top_k)
        return papers

    def save_index(self, path: str | Path) -> None:
        self.store.save(path)

    def load_index(self, path: str | Path) -> None:
        self.store.load(path)

    @property
    def index_size(self) -> int:
        return self.store.size
