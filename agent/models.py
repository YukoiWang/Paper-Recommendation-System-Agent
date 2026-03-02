"""Data models for papers, user profile, and recommendation results."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    published: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0

    def text_for_embedding(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        return ". ".join(parts) if parts else ""


@dataclass
class UserProfile:
    user_id: str
    interest_vector: Optional[np.ndarray] = field(default=None, repr=False)
    interest_text: str = ""
    liked_items: List[str] = field(default_factory=list)
    followed_authors: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    special_requirements: List[str] = field(default_factory=list)  # from user feedback
    # 用户学术/专业水平，用于 QA 调整语气：beginner | intermediate | expert | researcher
    expertise_level: Optional[str] = None


def is_profile_sufficient(profile: Optional[UserProfile]) -> bool:
    """True if profile has enough info for retrieval (interests or preferences)."""
    if profile is None:
        return False
    has_interests = bool(profile.interest_text or profile.interest_vector is not None)
    has_prefs = bool(profile.followed_authors or profile.preferred_categories or profile.liked_items)
    return has_interests or has_prefs


@dataclass
class RecommendationResult:
    user_id: str
    recommended_papers: List[Paper] = field(default_factory=list)
    recall_stats: Dict[str, int] = field(default_factory=dict)

    def summary(self, top_n: int = 10) -> str:
        lines = [
            f"User: {self.user_id}",
            f"Recall stats: {self.recall_stats}",
            f"Top-{min(top_n, len(self.recommended_papers))} recommendations:",
        ]
        for i, p in enumerate(self.recommended_papers[:top_n]):
            lines.append(f"  {i+1:2d}. [score={p.score:.4f}] {p.title}")
            if p.authors:
                lines.append(f"      Authors: {', '.join(p.authors[:3])}")
            if p.categories:
                lines.append(f"      Categories: {', '.join(p.categories)}")
        return "\n".join(lines)
