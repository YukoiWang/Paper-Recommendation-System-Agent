"""Retrieval agent: query -> HyDE (optional) + vector + BM25 hybrid, RRF fusion. QA path: no history filter."""
from __future__ import annotations
import json
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
        self._use_bm25 = use_bm25 and (BM25Okapi is not None)
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

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return " ".join(str(v).strip() for v in value if str(v).strip())
        return str(value).strip()

    @classmethod
    def _extract_json_records(cls, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in ("papers", "items", "data", "rows", "documents"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
        raise ValueError("Unsupported JSON format: expected a list or a dict containing papers/items/data.")

    def _paper_from_record(self, record: Dict[str, Any], paper_id: str) -> Paper:
        authors_raw = record.get("authors", [])
        if isinstance(authors_raw, str):
            authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        elif isinstance(authors_raw, list):
            authors = [str(a).strip() for a in authors_raw if str(a).strip()]
        else:
            authors = []

        categories_raw = record.get("categories", record.get("subjects", []))
        if isinstance(categories_raw, str):
            categories = [c.strip() for c in categories_raw.replace(",", " ").split() if c.strip()]
        elif isinstance(categories_raw, list):
            categories = [str(c).strip() for c in categories_raw if str(c).strip()]
        else:
            categories = []

        published = self._coerce_text(
            record.get("published")
            or record.get("update_date")
            or record.get("submission_date")
            or record.get("publication_date")
            or record.get("year")
        )

        return Paper(
            paper_id=paper_id,
            title=self._coerce_text(record.get("title")),
            abstract=self._coerce_text(record.get("abstract")),
            authors=authors[:10],
            categories=categories,
            published=published,
        )

    def _build_bm25(self, ids: List[str], texts: List[str]) -> None:
        if not self._use_bm25 or BM25Okapi is None or not ids:
            return
        corpus = [self._tokenize(t) for t in texts]
        self._bm25_index = BM25Okapi(corpus)
        self._bm25_ids = ids
        logger.info("BM25 index built: %s docs", len(ids))

    def init_bm25_from_json(
        self,
        json_path: str | Path,
        id_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        populate_cache: bool = True,
    ) -> int:
        """
        Initialize the in-memory BM25 index from a local JSON/JSONL paper dump.

        Expected record shape is flexible as long as it contains an ID field such as
        paper_id / id / arxiv_id and text fields such as title / abstract.

        Returns the number of indexed documents.
        """
        if not self._use_bm25 or BM25Okapi is None:
            logger.warning("BM25 disabled or rank_bm25 unavailable; skipping BM25 init from JSON.")
            return 0

        path = Path(json_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"JSON file not found: {path}")

        id_fields = id_fields or ["paper_id", "id", "arxiv_id"]
        text_fields = text_fields or ["title", "abstract"]

        if path.suffix.lower() == ".jsonl":
            records: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
        else:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            records = self._extract_json_records(payload)

        ids: List[str] = []
        texts: List[str] = []
        seen_ids = set()

        for record in records:
            paper_id = ""
            for field in id_fields:
                paper_id = self._coerce_text(record.get(field))
                if paper_id:
                    break
            if not paper_id or paper_id in seen_ids:
                continue

            parts = [self._coerce_text(record.get(field)) for field in text_fields]
            text = " [SEP] ".join(part for part in parts if part)
            if not text:
                continue

            ids.append(paper_id)
            texts.append(text)
            seen_ids.add(paper_id)

            if populate_cache and paper_id not in self._cache:
                self._cache[paper_id] = self._paper_from_record(record, paper_id)

            if limit is not None and len(ids) >= limit:
                break

        self._build_bm25(ids, texts)
        logger.info("Initialized BM25 from %s with %s docs", path, len(ids))
        return len(ids)

    def _ensure_chroma_cache_for_ids(self, ids: List[str]) -> None:
        if not ids or not self._use_chromadb or not isinstance(self.store, ChromaDBVectorStore):
            return
        missing_ids = [pid for pid in ids if pid not in self._cache]
        if not missing_ids:
            return
        try:
            for paper in self.store.get_papers_by_ids(missing_ids):
                pid = str(paper.get("paper_id", "")).strip()
                if not pid:
                    continue
                self._cache[pid] = Paper(
                    paper_id=pid,
                    title=paper.get("title", "") or "",
                    abstract=paper.get("abstract", "") or "",
                    authors=[],
                    categories=[],
                    published=str(paper.get("year", "") or ""),
                )
        except Exception as e:
            logger.warning("Failed to fetch Chroma metadata for %s ids: %s", len(missing_ids), e)

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

        fetch_k = max(top_k * 2, 40)
        if self._use_chromadb and isinstance(self.store, ChromaDBVectorStore):
            hits = self.store.search_with_metadata(query_vec, top_k=fetch_k)
            dense_results: List[Tuple[str, float]] = []
            for h in hits:
                pid = str(h["paper_id"])
                dense_results.append((pid, float(h["score"])))
                self._cache[pid] = Paper(
                    paper_id=pid,
                    title=h.get("title", "") or "",
                    abstract=h.get("abstract", "") or "",
                    authors=[],
                    categories=[],
                    published=str(h.get("year", "") or ""),
                    score=float(h["score"]),
                )
        else:
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

        if self._use_chromadb and isinstance(self.store, ChromaDBVectorStore):
            self._ensure_chroma_cache_for_ids([pid for pid, _ in fused])

        papers = []
        for pid, score in fused:
            paper = self._cache.get(pid)
            if paper:
                papers.append(Paper(
                    paper_id=paper.paper_id, title=paper.title, abstract=paper.abstract,
                    authors=paper.authors, categories=paper.categories, published=paper.published,
                    score=score,
                ))
        logger.info(
            "retrieve_by_query: %s papers (top_k=%s, hyde=%s, bm25=%s, chroma=%s)",
            len(papers), top_k, bool(hyde_document), bool(self._bm25_index), self._use_chromadb,
        )
        return papers

    def save_index(self, path: str | Path) -> None:
        self.store.save(path)

    def load_index(self, path: str | Path) -> None:
        self.store.load(path)

    @property
    def index_size(self) -> int:
        return self.store.size
