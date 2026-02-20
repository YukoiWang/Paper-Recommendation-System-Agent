"""Run LangGraph workflow. Entry point for paper recommendation."""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
if str(Path(__file__).resolve().parent.parent / "agent") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from langgraph_agents import build_workflow
from agent.models import UserProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph paper recommendation workflow")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="LLM API key")
    parser.add_argument("--top-k", type=int, default=10, help="Number of papers to return")
    parser.add_argument("--papers-path", default=None, help="Path to papers JSON for indexing")
    parser.add_argument("--query", default="", help="User query")
    parser.add_argument("--daily", action="store_true", help="Daily recommendation mode")
    parser.add_argument("--interest", default="machine learning", help="User interest text")
    args = parser.parse_args()

    from langgraph_agents import RetrievalAgent, RecallAgent, OnlineSearchAgent, RankAgent, PaperQAAgent

    retrieval = RetrievalAgent()
    recall = RecallAgent(retrieval_agent=retrieval)
    online = OnlineSearchAgent()
    rank = RankAgent()
    qa = PaperQAAgent(api_key=args.api_key)
    app = build_workflow(
        retrieval_agent=retrieval,
        recall_agent=recall,
        online_agent=online,
        rank_agent=rank,
        qa_agent=qa,
        api_key=args.api_key,
        default_top_k=args.top_k,
    )

    papers = []
    if args.papers_path:
        from agent.data_loader import load_from_lance_hf, ML_CATEGORIES
        papers = load_from_lance_hf(
            lance_path=args.papers_path,
            max_papers=5000,
            filter_categories=ML_CATEGORIES,
            prefer_recent=True,
        )
    else:
        from agent.data_loader import load_builtin_corpus
        papers = load_builtin_corpus()
    if papers:
        retrieval.index_papers(papers)
        logger.info("Indexed %s papers", retrieval.index_size)

    profile = UserProfile(user_id="user", interest_text=args.interest)
    initial_state = {
        "user_id": "user",
        "user_profile": profile,
        "user_query": args.query or ("daily recommendation" if args.daily else "recommend papers on transformers"),
        "is_daily_rec": args.daily,
        "top_k": args.top_k,
        "online_offline_fusion_ratio": 0.5,
        "history": [],
        "cited_papers": {},
    }

    async def run():
        result = await app.ainvoke(initial_state)
        return result

    final = asyncio.run(run())
    print("\n--- Response ---")
    print(final.get("response", "(no response)"))
    print("\n--- Final papers ---")
    for i, p in enumerate((final.get("final_papers") or [])[:args.top_k], 1):
        print(f"  {i}. {p.title[:60]}...")
    return final


if __name__ == "__main__":
    main()
