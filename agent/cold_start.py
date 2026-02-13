"""Cold start: resolve user vector and trending papers."""
from __future__ import annotations
import logging
from typing import List, Tuple
import numpy as np

from models import Paper, UserProfile

logger = logging.getLogger(__name__)

_DEFAULT_ML_TOPICS = [
    "deep learning neural network training optimization gradient descent",
    "large language model transformer attention mechanism pretraining",
    "reinforcement learning policy gradient reward model",
    "computer vision object detection image segmentation",
    "natural language processing text classification generation",
    "diffusion model generative AI image synthesis",
    "graph neural network representation learning",
    "federated learning privacy differential privacy",
]


def resolve_user_vector(user: UserProfile, embedder, corpus_papers: List[Paper]) -> Tuple[np.ndarray, bool]:
    """Return (user_vector, is_cold_start)."""
    if user.interest_vector is not None:
        vec = np.asarray(user.interest_vector, dtype=np.float32)
        logger.info("User %s: using interest_vector", user.user_id)
        return vec, False
    if user.interest_text:
        vec = embedder.encode(user.interest_text)
        logger.info("User %s: encoded interest_text", user.user_id)
        return vec, False
    logger.info("User %s: cold start, default ML topics", user.user_id)
    vecs = embedder.encode_batch(_DEFAULT_ML_TOPICS)
    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec.astype(np.float32), True


def get_trending_papers(papers: List[Paper], count: int = 10) -> List[Paper]:
    """Return top papers by published date (newest first)."""
    sorted_papers = sorted(papers, key=lambda p: p.published or "", reverse=True)
    return sorted_papers[:count]
