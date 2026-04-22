"""
Build training triples and a deduplicated passage corpus from MedHop.

Produces:
  data/corpus.json            {passage_id: text}  — all unique passages
  data/train_triples.json     list of triple dicts for training
  data/val_triples.json       list of triple dicts for evaluation

Triple dict schema:
  id         — MedHop example id
  query      — question string ("interacts_with DB00773?")
  answer     — answer drug-id string ("DB00072")
  positives  — [passage_ids that contain the answer string]
  supports   — [all passage_ids for this example]

Usage:
  python data_prep.py
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"


def _pid(text: str) -> str:
    """Deterministic 16-char hex id for a passage (deduplicates identical passages)."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def build_corpus(splits) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for split in splits:
        for ex in split:
            for passage in ex["supports"]:
                corpus[_pid(passage)] = passage
    return corpus


def build_triples(split, corpus_inv: dict[str, str]) -> list[dict]:
    """
    corpus_inv maps passage text -> passage_id.
    For each example, positives are passages that literally contain the answer drug-id.
    If none do (answer is implicit), fall back to the top-3 passages by position —
    MedHop curators place the most relevant passages first.
    """
    triples = []
    n_fallback = 0
    for ex in tqdm(split, desc="Building triples"):
        support_pids = [corpus_inv[p] for p in ex["supports"] if p in corpus_inv]
        answer = ex["answer"].strip()

        positives = [pid for pid, p in zip(support_pids, ex["supports"]) if answer in p]
        if not positives:
            positives = support_pids[:3]
            n_fallback += 1

        triples.append(
            {
                "id": ex["id"],
                "query": ex["query"],
                "answer": answer,
                "positives": positives,
                "supports": support_pids,
            }
        )

    pct = 100 * n_fallback / max(len(triples), 1)
    print(f"  positional fallback used for {n_fallback}/{len(triples)} ({pct:.1f}%) examples")
    return triples


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading MedHop …")
    ds = load_dataset("qangaroo", "medhop", verification_mode="no_checks")
    train_split = ds["train"]
    val_split = ds["validation"]

    print("Building corpus …")
    corpus = build_corpus([train_split, val_split])
    corpus_inv = {v: k for k, v in corpus.items()}
    print(f"  {len(corpus)} unique passages")

    print("Building training triples …")
    train_triples = build_triples(train_split, corpus_inv)

    print("Building validation triples …")
    val_triples = build_triples(val_split, corpus_inv)

    with open(DATA_DIR / "corpus.json", "w") as f:
        json.dump(corpus, f)
    with open(DATA_DIR / "train_triples.json", "w") as f:
        json.dump(train_triples, f, indent=2)
    with open(DATA_DIR / "val_triples.json", "w") as f:
        json.dump(val_triples, f, indent=2)

    print(f"\nSaved to {DATA_DIR}/")
    print(f"  corpus.json          — {len(corpus)} passages")
    print(f"  train_triples.json   — {len(train_triples)} examples")
    print(f"  val_triples.json     — {len(val_triples)} examples")


if __name__ == "__main__":
    main()
