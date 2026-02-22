"""Blackboard: shared state for multi-agent coordination.

Stores inputs/outputs of all agents: user profile, query, history, retrieval,
online search, fusion ratio, ranking, top-k, feedback, etc.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.models import Paper, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class BlackboardMessage:
    """Single message in conversation history."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Blackboard:
    """
    Shared message board for agent coordination.
    All agents read/write through this single source of truth.
    """

    # --- User & Session ---
    user_id: str = ""
    user_profile: Optional[UserProfile] = None
    conversation_id: str = ""

    # --- Query & Intent ---
    user_query: str = ""
    qa_intent: str = ""  # recommend, qa, compare, summarize, explain, general
    is_daily_rec: bool = False  # True = daily recommendation trigger, False = user active query

    # --- Conversation ---
    history: List[BlackboardMessage] = field(default_factory=list)
    cited_papers: Dict[str, Paper] = field(default_factory=dict)  # [1]->Paper, [2]->Paper

    # --- Retrieval / Recall ---
    recall_result: List[Paper] = field(default_factory=list)  # offline vector/rule/itemcf merge
    recall_stats: Dict[str, int] = field(default_factory=dict)

    # --- Online Search ---
    online_search_result: List[Paper] = field(default_factory=list)
    online_search_ids: List[str] = field(default_factory=list)  # IDs from Arxiv API

    # --- Fusion ---
    online_offline_fusion_ratio: float = 0.5  # 0=all offline, 1=all online
    fused_candidates: List[Paper] = field(default_factory=list)  # merged online+offline

    # --- Ranking ---
    ranked_papers: List[Paper] = field(default_factory=list)
    top_k: int = 10

    # --- Final Output ---
    final_papers: List[Paper] = field(default_factory=list)  # top_k after rank

    # --- User Feedback ---
    user_feedback: str = ""  # latest user feedback on recommendation/answer
    feedback_timestamp: float = 0.0

    # --- Planner / QA Coordination ---
    needs_profile_clarification: bool = False  # QA should proactively ask interests
    history_contains_papers: bool = True  # whether conversation already has paper context
    prefer_latest_papers: bool = False  # user wants newest papers -> boost online, recency
    profile_updated_from_feedback: bool = False  # QA updated profile from feedback

    # --- Misc ---
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str) -> BlackboardMessage:
        import time
        msg = BlackboardMessage(role="user", content=content, timestamp=time.time())
        self.history.append(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> BlackboardMessage:
        import time
        msg = BlackboardMessage(
            role="assistant", content=content, timestamp=time.time(),
            metadata=metadata or {},
        )
        self.history.append(msg)
        return msg

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def trim_history(self, max_turns: int = 20, max_chars: int = 60000):
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]
        total = sum(len(m.content) for m in self.history)
        while total > max_chars and len(self.history) > 2:
            removed = self.history.pop(0)
            total -= len(removed.content)

    def clear_conversation(self):
        self.history.clear()
        self.cited_papers.clear()

    def clear_turn_state(self):
        """Clear per-turn outputs (keep profile, history)."""
        self.recall_result.clear()
        self.recall_stats.clear()
        self.online_search_result.clear()
        self.online_search_ids.clear()
        self.fused_candidates.clear()
        self.ranked_papers.clear()
        self.final_papers.clear()
        self.user_feedback = ""
        self.error = None

    def snapshot(self) -> Dict[str, Any]:
        """For logging/debug: snapshot key fields."""
        return {
            "user_id": self.user_id,
            "query": self.user_query,
            "intent": self.qa_intent,
            "top_k": self.top_k,
            "fusion_ratio": self.online_offline_fusion_ratio,
            "recall_count": len(self.recall_result),
            "online_count": len(self.online_search_result),
            "ranked_count": len(self.ranked_papers),
            "final_count": len(self.final_papers),
        }
