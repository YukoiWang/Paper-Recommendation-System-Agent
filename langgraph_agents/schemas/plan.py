"""Pydantic work order and evidence pack (design §3.3 / §9)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

Intent = Literal[
    "chitchat", "meta", "feedback", "daily", "recommend",
    "explain", "compare", "survey", "factoid", "followup",
]
IntentSource = Literal["rule", "embedding", "llm"]
NextAgent = Literal["writer", "recommend", "researcher"]


class Budget(BaseModel):
    max_steps: int = 8
    max_search: int = 4
    max_fulltext: int = 2
    max_replan: int = 1


class SuccessCriteria(BaseModel):
    type: str = "explain"
    min_papers: int = 3
    min_papers_per_entity: int = 2
    min_chunks_rel_ge_3: int = 3
    min_chunks_rel_ge_3_per_entity: int = 2
    top_k: int = 5
    min_rerank_score: float = 0.0


class WorkOrder(BaseModel):
    intent: str = "explain"
    intent_source: str = "rule"
    confidence: float = 1.0
    entities: Dict[str, List[str]] = Field(default_factory=lambda: {"methods": []})
    slots: Dict[str, Any] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    next_agent: str = "writer"
    playbook_id: str = "explain_v1"
    budget: Budget = Field(default_factory=Budget)
    topic: str = ""
    referent_paper_ids: List[str] = Field(default_factory=list)
    need_sota: bool = False
    reasoning: str = ""

    def to_state_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()  # pydantic v1


class EvidenceChunk(BaseModel):
    chunk_id: str
    paper_id: str
    cite_n: int = 1
    text: str = ""
    section: str = "abstract"
    relevance: float = 0.0
    relevance_reason: str = ""


class EvidencePack(BaseModel):
    query: str = ""
    intent: str = "explain"
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[EvidenceChunk] = Field(default_factory=list)
    coverage: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    open_gaps: List[str] = Field(default_factory=list)
    tool_trace_summary: List[str] = Field(default_factory=list)
