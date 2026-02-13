"""Text embedders: TF-IDF + SVD or SentenceTransformer."""
from __future__ import annotations
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class TfidfEmbedder:
    """TF-IDF + TruncatedSVD + L2 normalization. No GPU required."""

    def __init__(self, dim: int = 256, max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer

        self.dim = dim
        self._fitted = False
        self._tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, dtype=np.float32,
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=42)
        self._norm = Normalizer(norm="l2")
        logger.info("TfidfEmbedder created: dim=%s", dim)

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        tfidf_matrix = self._tfidf.fit_transform(texts)
        actual_dim = min(self.dim, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        if actual_dim < self.dim:
            logger.warning("Reducing SVD dim %s -> %s", self.dim, actual_dim)
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            self.dim = actual_dim
        svd_matrix = self._svd.fit_transform(tfidf_matrix)
        self._norm.fit(svd_matrix)
        self._fitted = True
        explained = self._svd.explained_variance_ratio_.sum()
        logger.info("Fit complete. Explained variance: %.2%%, dim=%s", explained, self.dim)
        return self

    def encode(self, text: str) -> np.ndarray:
        self._check_fitted()
        tfidf = self._tfidf.transform([text])
        svd = self._svd.transform(tfidf)
        return self._norm.transform(svd)[0].astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        self._check_fitted()
        tfidf = self._tfidf.transform(texts)
        svd = self._svd.transform(tfidf)
        return self._norm.transform(svd).astype(np.float32)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder not fitted. Call embedder.fit(corpus_texts) first.")


class SentenceTransformerEmbedder:
    """Wrapper for sentence-transformers. Requires: pip install sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info("SentenceTransformerEmbedder: %s, dim=%s", model_name, self.dim)

    def fit(self, texts: List[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, text: str) -> np.ndarray:
        return np.array(self._model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.array(
            self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True),
            dtype=np.float32,
        )


def create_embedder(backend: str = "tfidf", dim: int = 256, model_name: str = "all-MiniLM-L6-v2"):
    if backend == "tfidf":
        return TfidfEmbedder(dim=dim)
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name=model_name)
    raise ValueError(f"Unknown backend: {backend}")
