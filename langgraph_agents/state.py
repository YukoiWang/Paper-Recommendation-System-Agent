"""LangGraph workflow state."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

import sys
from pathlib import Path
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.models import Paper, UserProfile


class WorkflowState(TypedDict, total=False):
    """State passed between LangGraph nodes. All fields optional for incremental updates."""
    # --- User & Session ---
    user_id: str
    user_profile: Optional[UserProfile]
    conversation_id: str

    # --- Query & Intent (set by Planner only) ---
    user_query: str
    user_feedback: str
    qa_intent: str
    is_daily_rec: bool
    needs_profile_clarification: bool
    prefer_latest_papers: bool

    # --- Conversation ---
    history: List[Dict[str, Any]]
    cited_papers: Dict[str, Paper]

    # --- Retrieval (query -> vector only) ---
    retrieval_result: List[Paper]

    # --- Recall (multi-path merge) ---
    online_search_result: List[Paper]
    recall_result: List[Paper]
    recall_stats: Dict[str, int]
    fused_candidates: List[Paper]
    online_offline_fusion_ratio: float

    # --- Ranking ---
    ranked_papers: List[Paper]
    final_papers: List[Paper]
    top_k: int

    # --- Plan (set by Planner) ---
    plan: Optional[Dict[str, Any]]

    # --- Output ---
    response: str
    error: Optional[str]
