"""
Orchestration layer: aggregate per-query metrics into mean scores, and
merge run results into the shared top-level results.json.

This is the module every component imports — retriever, reranker, router,
and Person D's end-to-end evaluator all write to the same file.
"""
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .metrics import recall_at_k, ndcg_at_k, map_score, reciprocal_rank

# results.json lives at the repo root, one level above eval_harness/
REPO_ROOT = Path(__file__).parent.parent
RESULTS_FILE = REPO_ROOT / "results.json"


def evaluate_rankings(
    rankings: dict[str, list[str]],
    relevants: dict[str, set[str]],
    k_values: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    Aggregate retrieval metrics across queries.

    Args:
        rankings:  {query_id: [passage_id, ...]}  — best-first
        relevants: {query_id: {passage_id, ...}}  — ground-truth set
        k_values:  cutoffs for Recall@K and nDCG@K

    Returns:
        Dict with keys recall@K, ndcg@K for each K, plus "map" and "mrr".
        Queries whose relevant set is empty are skipped (not counted as 0).
    """
    buckets: dict[str, list[float]] = {
        f"{m}@{k}": [] for m in ("recall", "ndcg") for k in k_values
    }
    buckets["map"] = []
    buckets["mrr"] = []

    for qid, retrieved in rankings.items():
        relevant = relevants.get(qid, set())
        if not relevant:
            continue
        for k in k_values:
            buckets[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            buckets[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))
        buckets["map"].append(map_score(retrieved, relevant))
        buckets["mrr"].append(reciprocal_rank(retrieved, relevant))

    return {
        key: float(np.mean(vals)) if vals else 0.0
        for key, vals in buckets.items()
    }


def append_to_leaderboard(
    component: str,
    run_name: str,
    metrics: dict,
    results_file: Path = RESULTS_FILE,
) -> None:
    """
    Merge one run's metrics into results.json.  Safe across components:
    loads the existing file, sets results[component][run_name] = metrics,
    and writes it back.  Never clobbers entries from other teammates.

    Example:
        append_to_leaderboard("reranker", "ms_marco_zero_shot",
                              {"recall@5": 0.52, ...})
    """
    existing: dict = {}
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    existing.setdefault(component, {})
    existing[component][run_name] = metrics

    with open(results_file, "w") as f:
        json.dump(existing, f, indent=2)


def print_leaderboard(results_file: Path = RESULTS_FILE) -> None:
    """Pretty-print all component results from results.json."""
    if not results_file.exists():
        print(f"No results file at {results_file}")
        return

    with open(results_file) as f:
        data = json.load(f)

    for component, runs in data.items():
        print(f"\n=== {component} ===")
        if not isinstance(runs, dict):
            print(f"  {runs}")
            continue
        # Collect all metric names to format a rough table
        for run_name, metrics in runs.items():
            if not isinstance(metrics, dict):
                print(f"  {run_name}: {metrics}")
                continue
            pieces = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    pieces.append(f"{k}={v:.3f}")
                else:
                    pieces.append(f"{k}={v}")
            print(f"  {run_name:<30} {'  '.join(pieces)}")


if __name__ == "__main__":
    # Quick sanity: evaluate a tiny synthetic set
    rankings = {
        "q1": ["p1", "p2", "p3"],
        "q2": ["p5", "p4", "p6"],
    }
    relevants = {
        "q1": {"p2"},
        "q2": {"p4"},
    }
    m = evaluate_rankings(rankings, relevants, k_values=(1, 3))
    print("Synthetic metrics:")
    for k, v in m.items():
        print(f"  {k:<12} {v:.4f}")