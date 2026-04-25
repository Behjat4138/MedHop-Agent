"""
Pure metric functions.  No I/O, no state.  All operate on:
  retrieved: list[str]      ranked passage IDs, best-first
  relevant:  set[str]       ground-truth relevant passage IDs

Binary-relevance conventions match retriever/evaluate.py (Person A).
"""
import numpy as np


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item appears in top-k, else 0.0.

    Matches Person A's definition — used because MedHop typically has a single
    answer-containing passage per question, so binary recall == fractional recall.
    """
    return float(bool(set(retrieved[:k]) & set(relevant)))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@K."""
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, pid in enumerate(retrieved[:k])
        if pid in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def map_score(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision for a single query.  Mean across queries = MAP."""
    if not relevant:
        return 0.0
    hits, ap_sum = 0, 0.0
    for i, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            hits += 1
            ap_sum += hits / i
    return ap_sum / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of first relevant item, or 0 if none retrieved.  Mean = MRR."""
    for i, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def exact_match(prediction: str, gold: str) -> float:
    """Lowercased, whitespace-normalized string equality.  For answer evaluation."""
    def norm(s: str) -> str:
        return " ".join(s.lower().strip().split())
    return float(norm(prediction) == norm(gold))


if __name__ == "__main__":
    # Sanity check
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    relevant = {"p2", "p4"}
    print(f"recall@1  = {recall_at_k(retrieved, relevant, 1):.4f}  (expect 0.0000)")
    print(f"recall@2  = {recall_at_k(retrieved, relevant, 2):.4f}  (expect 1.0000)")
    print(f"ndcg@5    = {ndcg_at_k(retrieved, relevant, 5):.4f}")
    print(f"map       = {map_score(retrieved, relevant):.4f}")
    print(f"mrr       = {reciprocal_rank(retrieved, relevant):.4f}  (expect 0.5000)")
    print(f"em hit    = {exact_match('DB00072', 'db00072'):.4f}  (expect 1.0000)")
    print(f"em miss   = {exact_match('DB00072', 'DB00073'):.4f}  (expect 0.0000)")