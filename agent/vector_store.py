"""Vector store backends: in-memory (NumPy) or persistent (LanceDB). Same interface: add, search, get_vector, save, load, clear."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class NumpyVectorStore:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._id_list: List[str] = []
        self._id_set: Set[str] = set()
        logger.info("NumpyVectorStore created: dim=%s", dim)

    @property
    def size(self) -> int:
        return len(self._id_list)

    def add(self, ids: List[str], vectors: np.ndarray) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected (n, {self.dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"len(ids)={len(ids)} != rows={vectors.shape[0]}")
        mask = [i for i, pid in enumerate(ids) if pid not in self._id_set]
        if not mask:
            return 0
        new_ids = [ids[i] for i in mask]
        new_vecs = vectors[mask]
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        new_vecs = new_vecs / norms
        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])
        for pid in new_ids:
            self._id_list.append(pid)
            self._id_set.add(pid)
        return len(new_ids)

    def search(self, query: np.ndarray, top_k: int = 10,
               exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        if self._vectors is None or self.size == 0:
            return []
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        scores = self._vectors @ query
        ranked_idx = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_idx:
            pid = self._id_list[idx]
            if exclude_ids and pid in exclude_ids:
                continue
            results.append((pid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def get_vector(self, paper_id: str) -> Optional[np.ndarray]:
        if paper_id not in self._id_set:
            return None
        idx = self._id_list.index(paper_id)
        return self._vectors[idx].copy()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(path / "vectors.npy", self._vectors)
        with open(path / "id_map.json", "w") as f:
            json.dump(self._id_list, f)
        logger.info("Saved to %s (%s vectors)", path, self.size)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._vectors = np.load(path / "vectors.npy").astype(np.float32)
        with open(path / "id_map.json") as f:
            self._id_list = json.load(f)
        self._id_set = set(self._id_list)
        self.dim = self._vectors.shape[1]
        logger.info("Loaded from %s (%s vectors)", path, self.size)

    def clear(self) -> None:
        self._vectors = None
        self._id_list.clear()
        self._id_set.clear()


class LanceDBVectorStore:
    """Persistent vector store via LanceDB. Requires: pip install lancedb."""

    TABLE_NAME = "paper_vectors"

    def __init__(self, dim: int = 256, db_path: str = "./.lancedb_index"):
        try:
            import lancedb as _lancedb
        except ImportError as e:
            raise ImportError("LanceDBVectorStore requires lancedb: pip install lancedb") from e
        self.dim = dim
        self.db_path = str(db_path)
        self._lancedb = _lancedb
        self._db = _lancedb.connect(self.db_path)
        self._table = None
        self._id_set: Set[str] = set()
        self._vec_cache: Dict[str, np.ndarray] = {}
        try:
            if self.TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(self.TABLE_NAME)
                existing_df = self._table.to_pandas()
                self._id_set = set(existing_df["id"].tolist())
                for _, row in existing_df.iterrows():
                    vec = row["vector"]
                    if vec is not None:
                        self._vec_cache[row["id"]] = np.asarray(vec, dtype=np.float32)
                logger.info("Opened existing table: %s vectors", len(self._id_set))
        except Exception as e:
            logger.warning("Could not open table: %s", e)
        logger.info("LanceDBVectorStore: dim=%s, path=%s", dim, self.db_path)

    @property
    def size(self) -> int:
        return len(self._id_set)

    def add(self, ids: List[str], vectors: np.ndarray) -> int:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected (n, {self.dim}), got {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"len(ids)={len(ids)} != rows={vectors.shape[0]}")
        mask = [i for i, pid in enumerate(ids) if pid not in self._id_set]
        if not mask:
            return 0
        new_ids = [ids[i] for i in mask]
        new_vecs = vectors[mask]
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        new_vecs = new_vecs / norms
        data = [{"id": pid, "vector": vec.tolist()} for pid, vec in zip(new_ids, new_vecs)]
        if self._table is None:
            self._table = self._db.create_table(self.TABLE_NAME, data)
        else:
            self._table.add(data)
        for pid, vec in zip(new_ids, new_vecs):
            self._id_set.add(pid)
            self._vec_cache[pid] = vec
        logger.info("LanceDB added %s vectors (total: %s)", len(new_ids), self.size)
        return len(new_ids)

    def search(self, query: np.ndarray, top_k: int = 10,
               exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        if self._table is None or self.size == 0:
            return []
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0)
        try:
            results_raw = (
                self._table.search(query.tolist()).metric("cosine").limit(fetch_k).to_list()
            )
        except Exception as e:
            logger.warning("LanceDB search failed: %s", e)
            return self._brute_force_search(query, top_k, exclude_ids)
        results = []
        for row in results_raw:
            pid = row["id"]
            if exclude_ids and pid in exclude_ids:
                continue
            cosine_sim = 1.0 - float(row.get("_distance", 0.0))
            results.append((pid, cosine_sim))
            if len(results) >= top_k:
                break
        return results

    def _brute_force_search(self, query: np.ndarray, top_k: int,
                            exclude_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        if not self._vec_cache:
            return []
        scores = []
        for pid, vec in self._vec_cache.items():
            if exclude_ids and pid in exclude_ids:
                continue
            sim = float(np.dot(vec, query))
            scores.append((pid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_vector(self, paper_id: str) -> Optional[np.ndarray]:
        if paper_id not in self._id_set:
            return None
        vec = self._vec_cache.get(paper_id)
        return vec.copy() if vec is not None else None

    def save(self, path: str | Path) -> None:
        logger.info("LanceDB persisted at %s (%s vectors). save() is a no-op.", self.db_path, self.size)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.db_path = str(path)
        self._db = self._lancedb.connect(self.db_path)
        self._table = None
        self._id_set.clear()
        self._vec_cache.clear()
        try:
            if self.TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(self.TABLE_NAME)
                existing_df = self._table.to_pandas()
                self._id_set = set(existing_df["id"].tolist())
                for _, row in existing_df.iterrows():
                    vec = row["vector"]
                    if vec is not None:
                        self._vec_cache[row["id"]] = np.asarray(vec, dtype=np.float32)
                if self._id_set and existing_df.iloc[0]["vector"] is not None:
                    self.dim = len(existing_df.iloc[0]["vector"])
        except Exception as e:
            logger.warning("load(%s) failed: %s", path, e)
        logger.info("Loaded LanceDB from %s (%s vectors)", path, self.size)

    def clear(self) -> None:
        if self._table is not None:
            try:
                self._db.drop_table(self.TABLE_NAME)
            except Exception:
                pass
            self._table = None
        self._id_set.clear()
        self._vec_cache.clear()


def create_vector_store(
    backend: str = "numpy",
    dim: int = 256,
    db_path: str = "./.lancedb_index",
) -> Any:
    if backend == "numpy":
        return NumpyVectorStore(dim=dim)
    if backend == "lancedb":
        return LanceDBVectorStore(dim=dim, db_path=db_path)
    raise ValueError(f"Unknown backend: {backend}. Use 'numpy' or 'lancedb'.")
