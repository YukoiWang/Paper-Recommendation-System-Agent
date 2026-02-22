#!/usr/bin/env python3
"""Quick demo: load built-in papers, index, recommend for a user."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.models import UserProfile
from agent import RetrievalAgent
from data_loader import load_builtin_corpus


def main():
    papers = load_builtin_corpus()
    agent = RetrievalAgent(
        embedding_backend="tfidf",
        embedding_dim=256,
        vector_store_backend="numpy",
        top_k_final=10,
    )
    agent.index_papers(papers)
    print(f"Indexed {agent.index_size} papers")

    user = UserProfile(
        user_id="demo",
        interest_text="large language model efficient training compression",
        preferred_categories=["cs.CL", "cs.LG"],
    )
    result = agent.retrieve_for_user(user)
    print(f"Recall: {result.recall_stats}")
    print(result.summary(top_n=5))


if __name__ == "__main__":
    main()
