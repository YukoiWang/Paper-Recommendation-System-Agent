"""Run LangGraph workflow. Entry point for paper recommendation.

Usage:
  # Single query
  python -m langgraph_agents.run_workflow --query "recommend papers on LoRA"

  # Interactive chat
  python -m langgraph_agents.run_workflow --chat

  # With Lance dataset
  python -m langgraph_agents.run_workflow --chat --papers-path /path/to/lance
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

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

from langgraph_agents import build_workflow
from agent.models import UserProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _build_agents(api_key: str, top_k: int, use_chromadb: bool = False,
                   chromadb_path: str = "/tmp/chroma_db",
                   embedding_model: str = "BAAI/bge-base-en-v1.5"):
    from langgraph_agents import (
        RetrievalAgent, RecallAgent, OnlineSearchAgent,
        RankAgent, PaperQAAgent, PlannerAgent,
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
    app = build_workflow(
        retrieval_agent=retrieval,
        recall_agent=recall,
        online_agent=online,
        planner_agent=planner,
        rank_agent=rank,
        qa_agent=qa,
        api_key=api_key,
        default_top_k=top_k,
    )
    return app, retrieval


def _load_and_index(retrieval, papers_path: str = None, max_papers: int = 5000):
    if papers_path:
        from agent.data_loader import load_from_lance_hf, ML_CATEGORIES
        papers = load_from_lance_hf(
            lance_path=papers_path,
            max_papers=max_papers,
            filter_categories=ML_CATEGORIES,
            prefer_recent=True,
        )
    else:
        from agent.data_loader import load_builtin_corpus
        papers = load_builtin_corpus()

    if papers:
        use_precomputed = getattr(papers[0], "embedding", None) is not None
        if use_precomputed:
            retrieval.index_papers_precomputed(papers)
        else:
            retrieval.index_papers(papers)
    return papers


def _print_response(result: dict, top_k: int):
    print("\n" + "=" * 60)
    print(result.get("response", "(no response)"))
    papers = (result.get("final_papers") or [])[:top_k]
    if papers:
        print(f"\n--- Retrieved {len(papers)} papers ---")
        for i, p in enumerate(papers, 1):
            print(f"  {i}. {p.title[:70]}{'...' if len(p.title) > 70 else ''}")
    decision = result.get("planner_decision") or {}
    if decision:
        online_tag = " +online_search" if decision.get("do_online_search") else ""
        print(f"\n[Planner] route={decision.get('route', '?')}{online_tag}  "
              f"optimized_query='{(decision.get('optimized_query') or '')[:50]}'")
        ev = decision.get("retrieval_evaluation")
        if ev:
            print(f"[Planner] retrieval_quality={ev.get('quality', '?')}  {ev.get('reasoning', '')[:60]}")
    print("=" * 60 + "\n")


def run_single(app, profile: UserProfile, query: str, top_k: int, is_daily: bool):
    state = {
        "user_id": profile.user_id,
        "user_profile": profile,
        "user_query": query or ("daily recommendation" if is_daily else "recommend papers on transformers"),
        "is_daily_rec": is_daily,
        "top_k": top_k,
        "online_offline_fusion_ratio": 0.5,
        "history": [],
        "cited_papers": {},
    }
    result = asyncio.run(app.ainvoke(state))
    _print_response(result, top_k)
    return result


def _run_with_timeout(app, state: dict, timeout: float = 90.0):
    """Run workflow with a timeout to avoid indefinite hangs."""
    async def _invoke():
        return await asyncio.wait_for(app.ainvoke(state), timeout=timeout)
    return asyncio.run(_invoke())


def run_chat(app, profile: UserProfile, top_k: int):
    logging.getLogger("langgraph_agents.rank_agent").setLevel(logging.ERROR)
    logging.getLogger("agent.embedder").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    print("\n" + "=" * 60)
    print("  Paper QA System — Interactive Chat")
    print("  Commands: /quit  /new  /debug  /papers")
    print("=" * 60 + "\n")

    history = []
    cited_papers = {}
    conversation_state = None

    while True:
        try:
            line = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/q", "exit", "quit"):
            print("Bye!")
            break
        if line.lower() == "/new":
            history = []
            cited_papers = {}
            conversation_state = None
            print("--- New conversation started ---\n")
            continue
        if line.lower() == "/debug":
            print(f"  history turns: {len(history)}")
            print(f"  cited papers: {len(cited_papers)}")
            if conversation_state:
                import json
                print(f"  conversation_state: {json.dumps(conversation_state, ensure_ascii=False, default=str)[:200]}")
            print()
            continue
        if line.lower() == "/papers":
            if cited_papers:
                for ref, p in cited_papers.items():
                    print(f"  {ref} {p.title[:60]}")
            else:
                print("  (no papers cited yet)")
            print()
            continue

        state = {
            "user_id": profile.user_id,
            "user_profile": profile,
            "user_query": line,
            "user_feedback": "",
            "is_daily_rec": False,
            "top_k": top_k,
            "online_offline_fusion_ratio": 0.5,
            "history": history,
            "cited_papers": cited_papers,
            "conversation_state": conversation_state,
        }

        try:
            t0 = time.time()
            print("  (thinking...)")
            result = _run_with_timeout(app, state, timeout=90.0)
            elapsed = time.time() - t0

            _print_response(result, top_k)
            print(f"  ({elapsed:.1f}s)\n")

            history = result.get("history", history)
            cited_papers = result.get("cited_papers", cited_papers)
            conversation_state = result.get("conversation_state", conversation_state)

        except asyncio.TimeoutError:
            print("\n  Timeout: LLM did not respond within 90s. Try again or /new.\n")
        except KeyboardInterrupt:
            print("\n  Interrupted. Type /quit to exit or ask another question.\n")
        except Exception as e:
            print(f"\n  Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph paper recommendation workflow")
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", ""), help="LLM API key")
    parser.add_argument("--top-k", type=int, default=8, help="Number of papers to return")
    parser.add_argument("--papers-path", default=None, help="Path to papers Lance dataset for indexing")
    parser.add_argument("--max-papers", type=int, default=5000, help="Max papers to load")
    parser.add_argument("--query", default="", help="Single query (non-interactive)")
    parser.add_argument("--daily", action="store_true", help="Daily recommendation mode")
    parser.add_argument("--interest", default="machine learning", help="User interest text")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--chromadb", action="store_true", help="Use existing ChromaDB (108万 papers)")
    parser.add_argument("--chromadb-path", default="/tmp/chroma_db", help="Path to ChromaDB")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5", help="Embedding model for ChromaDB queries")
    parser.add_argument("--log-level", default="WARNING", help="Logging level for chat mode")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: set --api-key or DEEPSEEK_API_KEY env var (or .env file)")
        sys.exit(1)

    if args.chat:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    print("Building agents...")
    app, retrieval = _build_agents(
        args.api_key, args.top_k,
        use_chromadb=args.chromadb,
        chromadb_path=args.chromadb_path,
        embedding_model=args.embedding_model,
    )

    if args.chromadb:
        print(f"Using ChromaDB at {args.chromadb_path} ({retrieval.index_size} papers)")
    else:
        print("Loading & indexing papers...")
        t0 = time.time()
        papers = _load_and_index(retrieval, args.papers_path, args.max_papers)
        print(f"Indexed {retrieval.index_size} papers in {time.time() - t0:.1f}s")

    profile = UserProfile(user_id="user", interest_text=args.interest)

    if args.chat:
        run_chat(app, profile, args.top_k)
    else:
        run_single(app, profile, args.query, args.top_k, args.daily)


if __name__ == "__main__":
    main()
