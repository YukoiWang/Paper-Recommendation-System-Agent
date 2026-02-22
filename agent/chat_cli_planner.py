#!/usr/bin/env python3
"""Interactive CLI using Planner + Blackboard. Run from project root or agent/."""
import os

from dotenv import load_dotenv
_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_env)

import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.models import UserProfile
from blackboard import Blackboard
from retrieval_agent import RetrievalAgent
from qa_agent import PaperQAAgent
from rank_agent import RankAgent
from planner_agent import PlannerAgent
from online_agent import OnlineSearchAgent


def load_papers(args):
    if args.papers == "builtin":
        from data_loader import load_builtin_corpus
        return load_builtin_corpus()
    if args.papers == "lance":
        from data_loader import load_from_lance_hf, ML_CATEGORIES
        return load_from_lance_hf(
            lance_path=args.lance_path or None,
            max_papers=args.max_papers,
            filter_categories=ML_CATEGORIES,
            prefer_recent=not getattr(args, "no_prefer_recent", False),
        )
    raise ValueError(f"Unknown source: {args.papers}")


def main():
    p = argparse.ArgumentParser(description="ML Paper Chat (Planner + Blackboard)")
    p.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    p.add_argument("--papers", default="builtin", choices=["builtin", "lance"])
    p.add_argument("--max-papers", type=int, default=5000)
    p.add_argument("--lance-path", default="")
    p.add_argument("--no-prefer-recent", action="store_true")
    p.add_argument("--embedding", default="tfidf", choices=["tfidf", "sentence_transformer"])
    p.add_argument("--vector-store", default="numpy", choices=["numpy", "lancedb"])
    p.add_argument("--vector-store-path", default="./.lancedb_index")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    if not args.api_key:
        print("Error: set --api-key or DEEPSEEK_API_KEY")
        sys.exit(1)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    print("Loading papers...")
    t0 = time.time()
    papers = load_papers(args)
    print(f"Loaded {len(papers)} papers in {time.time() - t0:.1f}s")

    use_precomputed = papers and getattr(papers[0], "embedding", None) is not None
    if use_precomputed:
        dim = papers[0].embedding.shape[0]
        retrieval = RetrievalAgent(
            embedding_backend="sentence_transformer",
            embedding_dim=dim,
            model_name="BAAI/bge-base-en-v1.5",
            vector_store_backend=args.vector_store,
            vector_store_path=args.vector_store_path,
            top_k_final=args.top_k * 2,
        )
        retrieval.index_papers_precomputed(papers)
    else:
        retrieval = RetrievalAgent(
            embedding_backend=args.embedding,
            embedding_dim=256,
            vector_store_backend=args.vector_store,
            vector_store_path=args.vector_store_path,
            top_k_final=args.top_k * 2,
        )
        retrieval.index_papers(papers)

    print(f"Indexed {retrieval.index_size} papers.")

    rank = RankAgent(mode="llm", api_key=args.api_key)
    qa = PaperQAAgent(retrieval_agent=retrieval, api_key=args.api_key, top_k_context=args.top_k)
    try:
        online = OnlineSearchAgent(time_window_days=30, max_results=15)
    except Exception:
        online = None

    planner = PlannerAgent(
        retrieval_agent=retrieval,
        rank_agent=rank,
        qa_agent=qa,
        online_agent=online,
        default_top_k=args.top_k,
    )

    user = UserProfile(
        user_id="user",
        interest_text="machine learning",
        preferred_categories=["cs.LG", "cs.AI"],
    )
    bb = Blackboard(
        user_id=user.user_id,
        user_profile=user,
        conversation_id="",
        top_k=args.top_k,
    )

    print("\nPlanner+Blackboard Chat. /quit to exit, /new for new conv, /daily for daily rec.\n")

    while True:
        try:
            line = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/q", "exit"):
            break
        if line.lower() == "/new":
            bb.clear_conversation()
            bb.conversation_id = ""
            print("New conversation.")
            continue
        if line.lower() == "/daily":
            bb.user_query = ""
            bb.is_daily_rec = True
            planner.run(bb, is_daily=True)
            if bb.history:
                print(bb.history[-1].content)
            else:
                print("(No response)")
            print()
            continue

        bb.user_query = line
        bb.user_profile = user
        planner.run(bb)
        if bb.history:
            print(bb.history[-1].content)
        else:
            print("(No response)")
        print()


if __name__ == "__main__":
    main()
