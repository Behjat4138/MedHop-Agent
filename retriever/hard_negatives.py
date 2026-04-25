"""
Hard negative mining for contrastive retriever training.

Two strategies:
  1. BM25  — passages that score high for the query but don't contain the answer.
             Operates within each example's own support set (closed-domain).
  2. Dense — passages that score high under the current bi-encoder but aren't positives.
             Requires a trained Encoder (see train.py).  Call after at least one epoch.

Usage:
  python hard_negatives.py                        # BM25 (no model needed)
  python hard_negatives.py --dense checkpoints/epoch_1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"


def _tok(text: str) -> list[str]:
    return text.lower().split()


# ---------------------------------------------------------------------------
# BM25 hard negatives
# ---------------------------------------------------------------------------

def mine_bm25_negatives(
    triples: list[dict],
    corpus: dict[str, str],
    n_negatives: int = 5,
) -> list[dict]:
    """
    For each triple, rank its support passages by BM25 score against the query.
    Return the top passages that are NOT positives and do NOT contain the answer.
    Adds / overwrites the 'hard_negatives' field in-place.
    """
    for triple in tqdm(triples, desc="BM25 hard negatives"):
        support_pids = triple["supports"]
        support_texts = [corpus[pid] for pid in support_pids if pid in corpus]

        if len(support_texts) < 2:
            triple["hard_negatives"] = []
            continue

        bm25 = BM25Okapi([_tok(t) for t in support_texts])
        scores = bm25.get_scores(_tok(triple["query"]))

        positive_set = set(triple["positives"])
        answer = triple["answer"]

        ranked = sorted(
            zip(support_pids[: len(scores)], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        hard_negs: list[str] = []
        for pid, _ in ranked:
            if pid in positive_set:
                continue
            if answer in corpus.get(pid, ""):
                continue
            hard_negs.append(pid)
            if len(hard_negs) >= n_negatives:
                break

        triple["hard_negatives"] = hard_negs

    return triples


# ---------------------------------------------------------------------------
# Dense hard negatives (requires a trained bi-encoder)
# ---------------------------------------------------------------------------

def mine_dense_negatives(
    triples: list[dict],
    corpus: dict[str, str],
    query_encoder,
    article_encoder,
    n_negatives: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> list[dict]:
    """
    Mine hard negatives using a bi-encoder.  For each query, find corpus passages
    that score high under the current model but are not true positives.
    Adds / overwrites the 'dense_hard_negatives' field in-place.
    """
    import torch
    from torch.nn.functional import normalize

    corpus_pids = list(corpus.keys())
    corpus_texts = [corpus[pid] for pid in corpus_pids]

    # Pre-encode the full corpus once
    print("Encoding corpus for dense hard negative mining …")
    passage_embs = article_encoder.encode(
        corpus_texts, batch_size=batch_size, device=device
    )  # [N, D]
    passage_embs = normalize(passage_embs.cpu(), dim=-1)

    article_encoder.eval()
    query_encoder.eval()

    for i in tqdm(range(0, len(triples), batch_size), desc="Dense hard negatives"):
        batch = triples[i : i + batch_size]
        queries = [t["query"] for t in batch]

        with torch.no_grad():
            q_embs = query_encoder.encode(queries, batch_size=batch_size, device=device)
            q_embs = normalize(q_embs.cpu(), dim=-1)

        sims = (q_embs @ passage_embs.T).numpy()  # [B, N]

        for j, triple in enumerate(batch):
            positive_set = set(triple["positives"])
            answer = triple["answer"]
            ranked_idx = np.argsort(sims[j])[::-1]

            dense_negs: list[str] = []
            for idx in ranked_idx:
                pid = corpus_pids[idx]
                if pid in positive_set:
                    continue
                if answer in corpus.get(pid, ""):
                    continue
                dense_negs.append(pid)
                if len(dense_negs) >= n_negatives:
                    break
            triple["dense_hard_negatives"] = dense_negs

    return triples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense",
        metavar="CHECKPOINT",
        default=None,
        help="Path to a trained checkpoint dir; if omitted only BM25 negatives are mined.",
    )
    parser.add_argument("--n_neg", type=int, default=5)
    args = parser.parse_args()

    print("Loading data …")
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)
    with open(DATA_DIR / "train_triples.json") as f:
        triples = json.load(f)

    print(f"Mining BM25 hard negatives for {len(triples)} examples …")
    triples = mine_bm25_negatives(triples, corpus, n_negatives=args.n_neg)

    if args.dense:
        from train import Encoder
        ckpt = Path(args.dense)
        print(f"Loading encoders from {ckpt} for dense mining …")
        q_enc = Encoder(str(ckpt / "query_encoder"))
        a_enc = Encoder(str(ckpt / "article_encoder"))
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q_enc.to(device)
        a_enc.to(device)
        triples = mine_dense_negatives(
            triples, corpus, q_enc, a_enc, n_negatives=args.n_neg, device=device
        )

    n_with = sum(1 for t in triples if t.get("hard_negatives"))
    print(f"{n_with}/{len(triples)} examples have BM25 hard negatives")

    out = DATA_DIR / "train_triples_with_negatives.json"
    with open(out, "w") as f:
        json.dump(triples, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
