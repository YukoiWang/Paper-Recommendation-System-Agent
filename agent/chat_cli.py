#!/usr/bin/env python3
"""Interactive CLI: load papers, index, chat with PaperQAAgent (DeepSeek API)."""
import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import UserProfile
from agent import RetrievalAgent
from qa_agent import PaperQAAgent


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
    p = argparse.ArgumentParser(description="ML Paper QA Chat")
    p.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", ""), help="DeepSeek API key")
    p.add_argument("--papers", default="builtin", choices=["builtin", "lance"], help="Paper source")
    p.add_argument("--max-papers", type=int, default=5000, help="Max papers (lance)")
    p.add_argument("--lance-path", default="", help="Local Lance dataset path")
    p.add_argument("--no-prefer-recent", action="store_true", help="Lance: do not sort by date")
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

    print(f"Indexed {retrieval.index_size} papers. Starting chat (/quit to exit).\n")

    qa = PaperQAAgent(retrieval_agent=retrieval, api_key=args.api_key, top_k_context=args.top_k)
    user = UserProfile(user_id="user", interest_text="machine learning", preferred_categories=["cs.LG", "cs.AI"])
    conv_id = None

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
            conv_id = None
            print("New conversation.")
            continue
        try:
            out = qa.chat(line, user=user, conversation_id=conv_id)
            conv_id = out["conversation_id"]
            print(out["response"])
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
