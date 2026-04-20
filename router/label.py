"""
Weak-supervision labeling via BM25 answer-surfacing score + median split.

MedHop is almost entirely multi-hop by design, so there's no clean 1-hop vs
multi-hop split. Instead we label relative difficulty: how well does BM25
surface the answer drug from the support passages in a single step?

    0 = "easier": BM25 answer score >= median  → single retrieval step likely sufficient
    1 = "harder": BM25 answer score <  median  → iterative multi-hop retrieval needed

Two-pass approach:
    Pass 1: compute the BM25 answer score for every training example
    Pass 2: split at the median to assign binary labels

Output: labels_train.json  { "MH_train_0": 1, "MH_train_1": 0, ... }

Usage: python label.py
"""

import json
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_answer_score(example: dict) -> float:
    """BM25 score of the top support passage when querying with the answer drug ID."""
    answer   = example["answer"]
    supports = example["supports"]
    tokenized_corpus = [_tokenize(s) for s in supports]
    bm25   = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores([answer.lower()])
    return float(scores.max())


def main():
    print("Loading MedHop train split...")
    dataset = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    train   = dataset["train"]

    # Pass 1: compute BM25 answer scores for all examples
    print("Pass 1: computing BM25 answer scores...")
    scores = []
    ids    = []
    for i, ex in enumerate(train):
        scores.append(bm25_answer_score(ex))
        ids.append(ex["id"])
        if (i + 1) % 200 == 0:
            print(f"  Scored {i+1}/{len(train)} ...")

    scores_arr = np.array(scores)
    median_score = float(np.median(scores_arr))
    print(f"\nBM25 answer score stats:")
    print(f"  min={scores_arr.min():.3f}  median={median_score:.3f}  max={scores_arr.max():.3f}")

    # Pass 2: median split → binary labels
    labels = {}
    easy_count = 0
    for qid, score in zip(ids, scores):
        label = 0 if score >= median_score else 1
        labels[qid] = label
        easy_count += (label == 0)

    hard_count = len(labels) - easy_count
    print(f"\nLabel distribution (median split at {median_score:.3f}):")
    print(f"  Easy (0 / single-step):  {easy_count}  ({100*easy_count/len(labels):.1f}%)")
    print(f"  Hard (1 / multi-step):   {hard_count}  ({100*hard_count/len(labels):.1f}%)")

    with open("labels_train.json", "w") as f:
        json.dump(labels, f, indent=2)
    print("\nSaved to labels_train.json")


if __name__ == "__main__":
    main()
