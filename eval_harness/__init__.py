"""Shared evaluation harness for the MedHop-Agent pipeline.

Person B's contribution — imported by retriever, reranker, router, and agent.
"""
from .metrics import (
    recall_at_k,
    ndcg_at_k,
    map_score,
    reciprocal_rank,
    exact_match,
)
from .runner import (
    evaluate_rankings,
    append_to_leaderboard,
    print_leaderboard,
)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "map_score",
    "reciprocal_rank",
    "exact_match",
    "evaluate_rankings",
    "append_to_leaderboard",
    "print_leaderboard",
]