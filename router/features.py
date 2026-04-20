"""
Feature extraction for the query router.

All features come from support passages + candidates because the MedHop
query text is always exactly "interacts_with DBXXXXX?" — no text signal there.

Public API used by train.py and router.py:
    extract_features(example: dict) -> np.ndarray  (shape: [5])

Feature vector layout (index → name):
    0  n_supports               number of support passages
    1  n_candidates             number of candidate answers
    2  top_bm25_score           BM25 score of best passage for the query drug ID
    3  bm25_score_gap           top_bm25_score minus second-best score (0 if only one passage)
    4  candidates_in_supports   fraction of candidates appearing in any support passage
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi


# Matches DrugBank IDs like DB00773
_DRUG_ID_RE = re.compile(r"DB\d{5}")


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization — fast and good enough for BM25."""
    return text.lower().split()


def _extract_query_drug(query: str) -> str:
    """Pull the DrugBank ID out of 'interacts_with DB00773?'."""
    match = _DRUG_ID_RE.search(query)
    return match.group(0) if match else ""


def extract_features(example: dict) -> np.ndarray:
    """
    example is one MedHop dict with keys: query, supports, candidates, answer, id.
    Returns a float32 array of shape (5,).
    """
    query_drug = _extract_query_drug(example["query"])
    supports   = example["supports"]     # list of strings
    candidates = example["candidates"]   # list of DrugBank ID strings

    n_supports   = len(supports)
    n_candidates = len(candidates)

    # BM25 over support passages, querying by the drug ID
    tokenized_corpus = [_tokenize(s) for s in supports]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores([query_drug.lower()])  # shape: (n_supports,)
    scores_sorted = sorted(scores, reverse=True)

    top_bm25_score = float(scores_sorted[0]) if scores_sorted else 0.0
    # Gap captures whether one passage clearly dominates vs. evidence is spread out
    bm25_score_gap = (
        float(scores_sorted[0] - scores_sorted[1])
        if len(scores_sorted) >= 2 else 0.0
    )

    # How many candidates appear anywhere in the supports (lowercased search)
    combined_supports = " ".join(supports).lower()
    hits = sum(1 for c in candidates if c.lower() in combined_supports)
    candidates_in_supports = hits / n_candidates if n_candidates > 0 else 0.0

    return np.array(
        [n_supports, n_candidates, top_bm25_score, bm25_score_gap, candidates_in_supports],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "n_supports",
    "n_candidates",
    "top_bm25_score",
    "bm25_score_gap",
    "candidates_in_supports",
]


if __name__ == "__main__":
    # Sanity check on a few examples
    from datasets import load_dataset

    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    print(f"Feature names: {FEATURE_NAMES}\n")
    for i in range(5):
        ex = dataset["train"][i]
        vec = extract_features(ex)
        print(f"Example {i}  query={ex['query']}")
        for name, val in zip(FEATURE_NAMES, vec):
            print(f"  {name:<28} {val:.4f}")
        print()
