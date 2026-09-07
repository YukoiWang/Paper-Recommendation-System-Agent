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
    retrieval_retry_count: int
    retrieval_insufficient: bool

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

    # --- Rerank expert loop (diagnostics; routing reads these) ---
    round: int
    last_rerank_score: float
    rerank_score: float
    rerank_diagnosis: str
    rerank_labels: List[str]
    rerank_primary: str
    rerank_suggestion: str
    rerank_confidence: float
    rerank_reasoning: str
    rerank_evidence: Dict[str, Any]
    qa_preference_score: float
    qa_rerank_count: int
    qa_needs_rerank: bool
    qa_feedback_for_rerank: str
    after_rank_dest: str
    # Eval alignment: when set, QA uses only this many papers and abstract chars (same as Ragas context).
    max_context_papers: Optional[int]
    max_context_abstract_chars: Optional[int]

    # --- Plan (set by Planner) ---
    plan: Optional[Dict[str, Any]]
    planner_decision: Optional[Dict[str, Any]]
    optimized_query: str
    final_query: str
    parent_query: str
    sub_queries: List[str]

    # --- Profile gathering ---
    profile_asked: bool
    profile_completed: bool

    # --- Conversation State Summary (written by QA agent each turn) ---
    conversation_state: Optional[Dict[str, Any]]

    # --- v2 work order (design doc) ---
    work_order: Optional[Dict[str, Any]]
    evidence_pack: Optional[Dict[str, Any]]
    critic_decision: Optional[str]
    critic_reason: Optional[str]
    critic_rewrite_count: int
    critic_more_count: int
    after_critic: str
    episode_id: str
    failure_type: Optional[str]
    recovery: Optional[str]
    researcher_trace: List[str]
    intent_source: str
    plan_ledger: Optional[Dict[str, Any]]
    progress_ledger: Optional[Dict[str, Any]]
    policy_versions: Optional[Dict[str, str]]
    needs_replan: bool
    replan_count: int
    writer_style_prompt: str
    daily_seen_ids: List[str]
    last_was_list: bool
