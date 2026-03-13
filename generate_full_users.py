#!/usr/bin/env python3
"""
generate_full_users.py - generate and enrich final user profiles in one step.

Supports two input modes:
1) Generate base users, then enrich them (default)
2) Load an existing users JSON file, then enrich it (via --users)

Examples:
  # Offline random generation + enrichment in one run
  python generate_full_users.py --num-users 100 --no-llm --papers lance --out users_100_full.json

  # LLM generation + enrichment
  python generate_full_users.py --num-users 50 --api-key sk-xxx --papers lance --out users_50_full.json

  # Enrich an existing users file
  python generate_full_users.py --users users_100.json --papers lance --out users_100_enriched.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from agent import RetrievalAgent
from enrich_users import enrich_users, load_papers
from generate_users import generate_users_with_llm, generate_users_without_llm
from user_profile_schema import RichUserProfile


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def resolve_api_key(cli_key: str) -> str:
    return cli_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")


def load_users(path: str) -> List[RichUserProfile]:
    with open(path, encoding="utf-8") as f:
        return [RichUserProfile.from_dict(d) for d in json.load(f)]


def save_users(users: List[RichUserProfile], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([u.to_dict() for u in users], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and enrich final user profiles in one command")

    # Input source: existing users or newly generated users
    parser.add_argument("--users", default="", help="Existing users JSON path. If set, skip generation step.")
    parser.add_argument("--num-users", type=int, default=100, help="Number of users to generate when --users is not set")
    parser.add_argument("--no-llm", action="store_true", help="Generate without LLM (offline random from schema)")
    parser.add_argument("--api-key", default="", help="API key for LLM generation")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation/enrichment randomness")
    parser.add_argument(
        "--generated-out",
        default="",
        help="Optional path to save generated users before enrichment",
    )

    # Enrichment parameters
    parser.add_argument("--papers", default="lance", choices=["builtin", "hf", "lance"])
    parser.add_argument("--max-papers", type=int, default=20000, help="Max papers loaded for enrichment")
    parser.add_argument("--num-liked", type=int, default=5, help="Number of liked papers per user")
    parser.add_argument("--num-authors", type=int, default=3, help="Number of followed authors per user")

    # Output
    parser.add_argument("--out", default="users_full.json", help="Final enriched users JSON path")
    args = parser.parse_args()

    # 1) Load or generate base users
    if args.users:
        print(f"  [1/4] Loading base users from {args.users} ...", flush=True)
        users = load_users(args.users)
        print(f"  [1/4] Loaded {len(users)} users.", flush=True)
    else:
        print(
            f"  [1/4] Generating base users (num={args.num_users}, mode={'no-llm' if args.no_llm else 'llm'}) ...",
            flush=True,
        )
        if args.no_llm:
            users = generate_users_without_llm(args.num_users, seed=args.seed)
        else:
            api_key = resolve_api_key(args.api_key)
            if not api_key:
                raise SystemExit(
                    "Need --api-key or OPENAI_API_KEY/DEEPSEEK_API_KEY for LLM generation. "
                    "Use --no-llm for offline mode."
                )
            users = generate_users_with_llm(args.num_users, api_key=api_key, base_url=args.base_url)
        print(f"  [1/4] Generated {len(users)} users.", flush=True)

        if args.generated_out:
            save_users(users, args.generated_out)
            print(f"  [1/4] Saved generated-only users to {args.generated_out}", flush=True)

    # 2) Load paper corpus
    print(f"  [2/4] Loading papers ({args.papers}, max={args.max_papers}) ...", flush=True)
    papers = load_papers(args.papers, args.max_papers)
    print(f"  [2/4] Loaded {len(papers)} papers.", flush=True)


    # 3) Build retrieval index
    print("  [3/4] Building retrieval index ...", flush=True)
    agent = RetrievalAgent(embedding_backend="tfidf", embedding_dim=256, vector_store_backend="numpy")
    agent.index_papers(papers)
    print(f"  [3/4] Indexed {agent.index_size} papers.", flush=True)

    # 4) Enrich users
    print(
        f"  [4/4] Enriching users (liked={args.num_liked}, authors={args.num_authors}) ...",
        flush=True,
    )
    users = enrich_users(
        users=users,
        papers=papers,
        agent=agent,
        num_liked=args.num_liked,
        num_authors=args.num_authors,
        seed=args.seed,
    )

    has_liked = sum(1 for u in users if u.liked_paper_ids)
    has_authors = sum(1 for u in users if u.followed_authors)
    avg_liked = sum(len(u.liked_paper_ids) for u in users) / max(1, len(users))
    avg_authors = sum(len(u.followed_authors) for u in users) / max(1, len(users))

    print("\n  Stats:", flush=True)
    print(f"    Users with liked_paper_ids: {has_liked}/{len(users)} (avg {avg_liked:.1f})", flush=True)
    print(f"    Users with followed_authors: {has_authors}/{len(users)} (avg {avg_authors:.1f})", flush=True)

    save_users(users, args.out)
    print(f"\n  Saved final enriched users to {args.out}", flush=True)


if __name__ == "__main__":
    main()
