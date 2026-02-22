"""System state for the agent pipeline: holds user, query, papers, and recommendation result."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.models import Paper, RecommendationResult, UserProfile


@dataclass
class SystemState:
    """
    State passed through the recommendation pipeline (recall -> optional rerank -> result).
    Aligns with models.Paper, UserProfile, RecommendationResult and agent flow.
    """
    user_id: str = ""
    query: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    candidate_papers: List[Paper] = field(default_factory=list)
    ranked_papers: List[Paper] = field(default_factory=list)
    result: Optional[RecommendationResult] = None
    recall_stats: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def with_candidates(self, papers: List[Paper], recall_stats: Optional[Dict[str, int]] = None) -> "SystemState":
        """Return a copy with candidate_papers (and optionally recall_stats) set."""
        return SystemState(
            user_id=self.user_id,
            query=self.query,
            user_profile=self.user_profile,
            candidate_papers=list(papers),
            ranked_papers=self.ranked_papers,
            result=self.result,
            recall_stats=recall_stats if recall_stats is not None else dict(self.recall_stats),
            error=self.error,
        )

    def with_ranked(self, papers: List[Paper]) -> "SystemState":
        """Return a copy with ranked_papers set."""
        return SystemState(
            user_id=self.user_id,
            query=self.query,
            user_profile=self.user_profile,
            candidate_papers=self.candidate_papers,
            ranked_papers=list(papers),
            result=self.result,
            recall_stats=self.recall_stats,
            error=self.error,
        )

    def with_result(self, result: RecommendationResult) -> "SystemState":
        """Return a copy with result set."""
        return SystemState(
            user_id=self.user_id,
            query=self.query,
            user_profile=self.user_profile,
            candidate_papers=self.candidate_papers,
            ranked_papers=self.ranked_papers,
            result=result,
            recall_stats=self.recall_stats,
            error=self.error,
        )

    def with_error(self, message: str) -> "SystemState":
        """Return a copy with error set."""
        return SystemState(
            user_id=self.user_id,
            query=self.query,
            user_profile=self.user_profile,
            candidate_papers=self.candidate_papers,
            ranked_papers=self.ranked_papers,
            result=self.result,
            recall_stats=self.recall_stats,
            error=message,
        )
