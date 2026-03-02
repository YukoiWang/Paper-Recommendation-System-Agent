"""Web UI for Paper QA System.

Usage:
  python3.10 langgraph_agents/web_app.py [--port 8000] [--papers-path PATH]

Opens a chat interface at http://localhost:8000
Logs: logs/state_log.jsonl, logs/planner_log.jsonl
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
if str(Path(__file__).resolve().parent.parent / "agent") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"))

from fastapi import FastAPI, Request, Response, Cookie
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent.models import UserProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("langgraph_agents.rank_agent").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized on startup)
# ---------------------------------------------------------------------------
app = FastAPI(title="Paper QA System")
workflow_app = None
retrieval_agent = None

LOG_DIR = Path(_root) / "logs"
STATE_LOG = LOG_DIR / "state_log.jsonl"
PLANNER_LOG = LOG_DIR / "planner_log.jsonl"

SESSION_TTL = 3600 * 6  # auto-expire sessions after 6 hours


class SessionData:
    __slots__ = (
        "profile", "history", "cited_papers", "conversation_state",
        "profile_asked", "profile_completed", "turn_counter", "last_active",
    )

    def __init__(self, sid: str):
        self.profile = UserProfile(user_id=sid, interest_text="machine learning")
        self.history: List[Dict[str, Any]] = []
        self.cited_papers: Dict[str, Any] = {}
        self.conversation_state: Optional[Dict[str, Any]] = None
        self.profile_asked: bool = False
        self.profile_completed: bool = False
        self.turn_counter: int = 0
        self.last_active: float = time.time()


sessions: Dict[str, SessionData] = {}


def _get_or_create_session(sid: str | None) -> tuple[str, SessionData]:
    """Return (session_id, session_data), creating a new session if needed."""
    now = time.time()
    # Evict expired sessions periodically
    if len(sessions) > 200:
        expired = [k for k, v in sessions.items() if now - v.last_active > SESSION_TTL]
        for k in expired:
            del sessions[k]

    if sid and sid in sessions:
        s = sessions[sid]
        s.last_active = now
        return sid, s

    new_sid = uuid.uuid4().hex[:12]
    s = SessionData(new_sid)
    sessions[new_sid] = s
    return new_sid, s


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)


def _serialize_state(state: dict) -> dict:
    """Convert state to JSON-serializable dict (strip embeddings, Paper objects)."""
    out = {}
    skip_keys: set = set()
    for k, v in state.items():
        if k in skip_keys:
            continue
        if k == "user_profile" and v is not None:
            out[k] = {
                "user_id": getattr(v, "user_id", ""),
                "interest_text": getattr(v, "interest_text", ""),
                "liked_items": getattr(v, "liked_items", []),
                "followed_authors": getattr(v, "followed_authors", []),
                "preferred_categories": getattr(v, "preferred_categories", []),
                "special_requirements": getattr(v, "special_requirements", []),
            }
            continue
        if isinstance(v, list) and v and hasattr(v[0], "paper_id"):
            out[k] = [{"paper_id": p.paper_id, "title": p.title, "score": getattr(p, "score", 0)}
                       for p in v[:15]]
        elif isinstance(v, dict) and v and any(hasattr(vv, "paper_id") for vv in v.values()):
            out[k] = {ref: {"paper_id": p.paper_id, "title": p.title} for ref, p in v.items()
                       if hasattr(p, "paper_id")}
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v, default=str)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)[:500]
        else:
            out[k] = str(v)[:300]
    return out


def log_state(turn: int, user_query: str, result: dict, elapsed: float):
    _ensure_log_dir()
    entry = {
        "turn": turn,
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "elapsed_s": round(elapsed, 2),
        "state": _serialize_state(result),
    }
    with open(STATE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def log_planner(turn: int, user_query: str, result: dict, elapsed: float):
    _ensure_log_dir()
    decision = result.get("planner_decision") or {}
    ev = decision.get("retrieval_evaluation")
    entry = {
        "turn": turn,
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "optimized_query": decision.get("optimized_query", ""),
        "route": decision.get("route", "?"),
        "response_style": decision.get("response_style", "recommend"),
        "do_online_search": decision.get("do_online_search", False),
        "reasoning": decision.get("reasoning", ""),
        "retrieval_quality": ev.get("quality", "") if ev else None,
        "retrieval_reasoning": ev.get("reasoning", "") if ev else None,
        "suggested_refined_query": ev.get("suggested_refined_query") if ev else None,
        "num_final_papers": len(result.get("final_papers") or []),
        "elapsed_s": round(elapsed, 2),
    }
    with open(PLANNER_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    top_k: int = 8


class ChatResponse(BaseModel):
    response: str
    papers: List[Dict[str, str]]
    planner_route: str
    planner_optimized_query: str
    planner_reasoning: str
    do_online_search: bool
    retrieval_quality: Optional[str]
    elapsed: float
    turn: int


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request, response: Response):
    sid = request.cookies.get("sid")
    sid, sess = _get_or_create_session(sid)
    response.set_cookie("sid", sid, max_age=SESSION_TTL, httponly=True, samesite="lax")

    state = {
        "user_id": sess.profile.user_id,
        "user_profile": sess.profile,
        "user_query": req.message,
        "user_feedback": "",
        "is_daily_rec": False,
        "top_k": req.top_k,
        "online_offline_fusion_ratio": 0.5,
        "history": sess.history,
        "cited_papers": sess.cited_papers,
        "conversation_state": sess.conversation_state,
        "profile_asked": sess.profile_asked,
        "profile_completed": sess.profile_completed,
    }

    t0 = time.time()
    try:
        result = await asyncio.wait_for(workflow_app.ainvoke(state), timeout=90.0)
    except asyncio.TimeoutError:
        return ChatResponse(
            response="Sorry, the request timed out (90s). Please try again.",
            papers=[], planner_route="TIMEOUT", planner_optimized_query="",
            planner_reasoning="Timeout", do_online_search=False,
            retrieval_quality=None, elapsed=90.0, turn=sess.turn_counter,
        )
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("Workflow error for query %r:\n%s", req.message, tb)
        elapsed = time.time() - t0
        return ChatResponse(
            response=f"Sorry, an internal error occurred: {type(exc).__name__}: {exc}",
            papers=[], planner_route="ERROR", planner_optimized_query="",
            planner_reasoning=str(exc)[:300], do_online_search=False,
            retrieval_quality=None, elapsed=round(elapsed, 2), turn=sess.turn_counter,
        )
    elapsed = time.time() - t0
    sess.turn_counter += 1

    sess.history = result.get("history", sess.history)
    sess.cited_papers = result.get("cited_papers", sess.cited_papers)
    sess.conversation_state = result.get("conversation_state", sess.conversation_state)
    sess.profile_asked = result.get("profile_asked", sess.profile_asked)
    sess.profile_completed = result.get("profile_completed", sess.profile_completed)
    updated_profile = result.get("user_profile")
    if updated_profile is not None:
        sess.profile = updated_profile

    try:
        log_state(sess.turn_counter, req.message, result, elapsed)
        log_planner(sess.turn_counter, req.message, result, elapsed)
    except Exception:
        logger.exception("Failed to write logs")

    decision = result.get("planner_decision") or {}
    ev = decision.get("retrieval_evaluation")
    papers = []
    for i, p in enumerate((result.get("final_papers") or [])[:req.top_k], 1):
        papers.append({
            "index": str(i),
            "title": p.title,
            "authors": ", ".join(p.authors[:3]) + ("..." if len(p.authors) > 3 else ""),
            "categories": ", ".join(p.categories),
            "published": p.published or "",
        })

    return ChatResponse(
        response=result.get("response", "(no response)"),
        papers=papers,
        planner_route=decision.get("route", "?"),
        planner_optimized_query=decision.get("optimized_query", ""),
        planner_reasoning=decision.get("reasoning", ""),
        do_online_search=decision.get("do_online_search", False),
        retrieval_quality=ev.get("quality") if ev else None,
        elapsed=round(elapsed, 2),
        turn=sess.turn_counter,
    )


@app.post("/api/new")
async def new_session(request: Request, response: Response):
    sid = request.cookies.get("sid")
    if sid and sid in sessions:
        del sessions[sid]
    new_sid = uuid.uuid4().hex[:12]
    sessions[new_sid] = SessionData(new_sid)
    response.set_cookie("sid", new_sid, max_age=SESSION_TTL, httponly=True, samesite="lax")
    return {"status": "ok"}


@app.get("/api/logs/state")
async def get_state_logs():
    if not STATE_LOG.exists():
        return JSONResponse([])
    lines = STATE_LOG.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines:
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return JSONResponse(entries)


@app.get("/api/logs/planner")
async def get_planner_logs():
    if not PLANNER_LOG.exists():
        return JSONResponse([])
    lines = PLANNER_LOG.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines:
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return JSONResponse(entries)


@app.get("/api/logs/clear")
async def clear_logs():
    for f in [STATE_LOG, PLANNER_LOG]:
        if f.exists():
            f.write_text("")
    return {"status": "cleared"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "chat_ui.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def init_workflow(api_key: str = "", papers_path: str = None,
                  max_papers: int = 5000, top_k: int = 8,
                  use_chromadb: bool = False,
                  chromadb_path: str = None,  # default: ~/chroma_db via argparse
                  embedding_model: str = "BAAI/bge-base-en-v1.5"):
    global workflow_app, retrieval_agent
    chromadb_path = chromadb_path or os.path.expanduser("~/chroma_db")
    from langgraph_agents import (
        build_workflow, RetrievalAgent, RecallAgent,
        OnlineSearchAgent, RankAgent, PaperQAAgent, PlannerAgent,
    )
    if use_chromadb:
        retrieval = RetrievalAgent(
            vector_store_backend="chromadb",
            embedding_backend="sentence_transformer",
            model_name=embedding_model,
            chromadb_path=chromadb_path,
        )
    else:
        retrieval = RetrievalAgent()

    recall = RecallAgent(retrieval_agent=retrieval)
    online = OnlineSearchAgent()
    planner = PlannerAgent(api_key=api_key)
    rank = RankAgent()
    qa = PaperQAAgent(api_key=api_key)
    workflow_app = build_workflow(
        retrieval_agent=retrieval, recall_agent=recall,
        online_agent=online, planner_agent=planner,
        rank_agent=rank, qa_agent=qa,
        api_key=api_key, default_top_k=top_k,
    )
    retrieval_agent = retrieval

    if not use_chromadb:
        if papers_path:
            from agent.data_loader import load_from_lance_hf, ML_CATEGORIES
            papers = load_from_lance_hf(
                lance_path=papers_path, max_papers=max_papers,
                filter_categories=ML_CATEGORIES, prefer_recent=True,
            )
        else:
            from agent.data_loader import load_builtin_corpus
            papers = load_builtin_corpus()
        if papers:
            precomputed = getattr(papers[0], "embedding", None) is not None
            if precomputed:
                retrieval.index_papers_precomputed(papers)
            else:
                retrieval.index_papers(papers)
    logger.info("Web app: %d papers available", retrieval.index_size)
    _ensure_log_dir()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Paper QA Web UI")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--papers-path", default=None)
    parser.add_argument("--max-papers", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chromadb", action="store_true", help="Use existing ChromaDB")
    parser.add_argument("--chromadb-path", default=os.path.expanduser("~/chroma_db"))
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set --api-key or DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    print("Initializing workflow...")
    init_workflow(api_key=args.api_key, papers_path=args.papers_path,
                  max_papers=args.max_papers, top_k=args.top_k,
                  use_chromadb=args.chromadb,
                  chromadb_path=args.chromadb_path,
                  embedding_model=args.embedding_model)
    print(f"Starting web UI at http://localhost:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
