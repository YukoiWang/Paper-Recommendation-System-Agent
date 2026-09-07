"""LangGraph agents: retrieval, recall, online search, planner, QA, rank."""
from langgraph_agents.state import WorkflowState
from langgraph_agents.retrieval_agent import RetrievalAgent
from langgraph_agents.recall_agent import RecallAgent
from langgraph_agents.online_search_agent import OnlineSearchAgent
from langgraph_agents.planner_agent import (
    PlannerAgent,
    ROUTE_NO_RETRIEVAL,
    ROUTE_RETRIEVE_LOCAL,
    ROUTE_NEED_CLARIFY,
    ROUTE_HANDLE_FEEDBACK,
)
from langgraph_agents.qa_agent import PaperQAAgent, LLMClient
from langgraph_agents.rank_agent import RankAgent


def build_workflow(*args, **kwargs):
    """Default: v2 graph (design doc). Use build_workflow_legacy for the old DAG."""
    from langgraph_agents.workflow_v2 import build_workflow_v2
    return build_workflow_v2(*args, **kwargs)


def build_workflow_legacy(*args, **kwargs):
    from langgraph_agents.workflow import build_workflow as _build
    return _build(*args, **kwargs)


__all__ = [
    "WorkflowState",
    "RetrievalAgent",
    "RecallAgent",
    "OnlineSearchAgent",
    "PlannerAgent",
    "PaperQAAgent",
    "RankAgent",
    "build_workflow",
    "build_workflow_legacy",
    "LLMClient",
    "ROUTE_NO_RETRIEVAL",
    "ROUTE_RETRIEVE_LOCAL",
    "ROUTE_NEED_CLARIFY",
    "ROUTE_HANDLE_FEEDBACK",
]
