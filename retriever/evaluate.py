"""
Retrieval evaluation on the MedHop validation set.

Metrics: Recall@K and nDCG@K for K in {1, 5, 10}.

Evaluation design (closed-domain, per-example):
  MedHop is a closed-domain benchmark: each question comes with ~30 curated
  support passages.  The retriever must RANK those passages so that the most
  relevant ones appear at the top.  Evaluating over a global 50k corpus is
  inappropriate here because:
    1. MedHop passages use drug names, not DrugBank IDs, so string-match
       relevance against the answer ID (e.g. "DB00072") almost never fires.
    2. Finding 1 relevant passage in 50k at K=10 is an unreasonably hard task.

  Instead we:
    - Treat ALL support passages for an example as the relevant set
      (MedHop curators selected them specifically for that question).
    - Ask: given the query, how well does the retriever rank these passages
      among themselves?  i.e. retrieve from each example's own passage pool.
  This mirrors how a pipeline actually uses the retriever (given a pre-fetched
  candidate set, rank it).

Models evaluated (in order):
  1. BM25               (lexical baseline)
  2. Vanilla MedCPT     (zero-shot biomedical dense)
  3. SBERT              (general-purpose dense)
  4. DPR                (general-purpose open-domain dense)
  5. Fine-tuned MedCPT  (our contribution — only if --checkpoint is given)

Results are merged into results.json under the "retriever" key so both the
router (Person B) and retriever (Person A) results live in the same file.

Usage:
  python evaluate.py --baselines_only
  python evaluate.py --checkpoint checkpoints/epoch_5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = Path(__file__).parent.parent / "results.json"
RESULTS_TXT  = Path(__file__).parent / "baseline_results.txt"


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Binary-relevance nDCG@K.
    Ideal DCG = sum of 1/log2(i+2) for i in range(min(|relevant|, k)).
    """
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, pid in enumerate(retrieved[:k])
        if pid in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-example closed-domain evaluation helpers
# ---------------------------------------------------------------------------

def _bm25_rank(query: str, support_pids: list[str], corpus: dict[str, str], k: int):
    """Rank a small passage pool with BM25; returns [(pid, score), …]."""
    from rank_bm25 import BM25Okapi
    texts = [corpus.get(pid, "") for pid in support_pids]
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    scores = bm25.get_scores(query.lower().split())
    top = np.argsort(scores)[::-1][:k]
    return [(support_pids[i], float(scores[i])) for i in top]


def _dense_rank(
    query: str,
    support_pids: list[str],
    corpus: dict[str, str],
    precomputed_embs: dict[str, "torch.Tensor"],
    query_encoder,
    device: str,
    k: int,
):
    """
    Rank a small passage pool with a bi-encoder.
    precomputed_embs: {pid: [D] tensor} — built once before the eval loop.
    """
    import torch
    from torch.nn.functional import normalize

    q_emb = query_encoder.encode([query], max_length=64, batch_size=1, device=device)
    q_emb = normalize(q_emb.cpu(), dim=-1)  # [1, D]

    pids_present = [p for p in support_pids if p in precomputed_embs]
    if not pids_present:
        return []
    p_embs = torch.stack([precomputed_embs[p] for p in pids_present])  # [N, D]
    scores = (q_emb @ p_embs.T).squeeze(0).numpy()
    top = np.argsort(scores)[::-1][:k]
    return [(pids_present[i], float(scores[i])) for i in top]


def _precompute_embs(
    corpus: dict[str, str],
    article_encoder,
    device: str,
    batch_size: int = 64,
) -> dict[str, "torch.Tensor"]:
    """Encode every corpus passage once; returns {pid: [D] tensor}."""
    import torch
    from torch.nn.functional import normalize

    pids = list(corpus.keys())
    texts = [corpus[p] for p in pids]
    embs = article_encoder.encode(texts, max_length=512, batch_size=batch_size, device=device)
    embs = normalize(embs.cpu(), dim=-1)
    return {pid: embs[i] for i, pid in enumerate(pids)}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever_type: str,
    val_triples: list[dict],
    corpus: dict[str, str],
    k_values: tuple[int, ...] = (1, 5, 10),
    *,
    # Dense-only arguments
    query_encoder=None,
    precomputed_embs: dict | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Closed-domain per-example evaluation.

    For each validation example:
      - Relevant set = ALL support passages (MedHop curates them for the question).
      - Retrieve from the example's own passage pool only.
      - Compute Recall@K and nDCG@K.

    retriever_type: "bm25" | "dense"
    """
    buckets: dict[str, list[float]] = {
        f"{metric}@{k}": []
        for metric in ("recall", "ndcg")
        for k in k_values
    }

    n_skipped = 0
    for triple in tqdm(val_triples, desc="  eval", leave=False):
        support_pids = [p for p in triple["supports"] if p in corpus]
        if not support_pids:
            continue

        # Relevant = passages that contain the answer drug-ID as a substring.
        # The retrieval pool is the example's own support passages (closed-domain).
        # We never set relevant = all supports because that makes every retrieval trivially correct.
        relevant = {
            pid for pid in support_pids
            if triple["answer"] in corpus[pid]
        }
        if not relevant:
            # This example has no answer-containing passage — skip it rather than
            # polluting metrics with a task that has no ground-truth signal.
            n_skipped += 1
            continue

        max_k = min(max(k_values), len(support_pids))

        try:
            if retriever_type == "bm25":
                results = _bm25_rank(triple["query"], support_pids, corpus, max_k)
            else:
                results = _dense_rank(
                    triple["query"], support_pids, corpus,
                    precomputed_embs, query_encoder, device, max_k,
                )
        except Exception as exc:
            print(f"  WARNING: {triple['id']}: {exc}", file=sys.stderr)
            continue

        retrieved = [pid for pid, _ in results]
        for k in k_values:
            buckets[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            buckets[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    n_evaluated = len(next(iter(buckets.values())))
    print(f"    evaluated {n_evaluated} examples  ({n_skipped} skipped — no answer in support set)")
    return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in buckets.items()}


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _table_lines(results: dict[str, dict], k_values: tuple[int, ...]) -> list[str]:
    recall_hdrs = "".join(f"  R@{k:<3d}" for k in k_values)
    ndcg_hdrs   = "".join(f"  nDCG@{k:<3d}" for k in k_values)
    header = f"{'Model':<25}{recall_hdrs}{ndcg_hdrs}"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for name, scores in results.items():
        row  = f"{name:<25}"
        row += "".join(f"  {scores.get(f'recall@{k}', 0):.3f}  " for k in k_values)
        row += "".join(f"  {scores.get(f'ndcg@{k}', 0):.3f}  " for k in k_values)
        lines.append(row)
    return lines


def print_table(results: dict[str, dict], k_values: tuple[int, ...]) -> None:
    print()
    for line in _table_lines(results, k_values):
        print(line)


def save_txt_report(
    results: dict[str, dict],
    k_values: tuple[int, ...],
    path: Path,
    checkpoint: str | None,
) -> None:
    """Write a human-readable results report to a plain-text file."""
    import datetime

    lines = [
        "MedHop Retrieval Evaluation",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Checkpoint: {checkpoint or 'none (baselines only)'}",
        f"Eval set: MedHop validation split (342 examples)",
        f"Eval mode: closed-domain — ranking within each question's own ~30 support passages",
        f"Relevant set: passages that contain the answer drug-ID as a substring",
        f"  (examples where no support passage contains the answer are excluded)",
        "",
    ]
    lines += _table_lines(results, k_values)
    lines += [
        "",
        "Per-model scores (full):",
    ]
    for name, scores in results.items():
        lines.append(f"  {name}:")
        for metric, val in sorted(scores.items()):
            lines.append(f"    {metric:<12} {val:.4f}")

    path.write_text("\n".join(lines) + "\n")
    print(f"Text report saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="DIR",
        help="Path to fine-tuned checkpoint (e.g. checkpoints/epoch_5); "
             "if omitted, fine-tuned MedCPT is skipped.",
    )
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help="Run only BM25 (no GPU / transformers needed).",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        metavar="K",
    )
    args = parser.parse_args()
    k_values = tuple(args.k_values)

    # ---- data ---------------------------------------------------------------
    print("Loading data …")
    with open(DATA_DIR / "corpus.json") as f:
        corpus: dict[str, str] = json.load(f)
    with open(DATA_DIR / "val_triples.json") as f:
        val_triples: list[dict] = json.load(f)

    print(f"  corpus:     {len(corpus)} passages")
    print(f"  val set:    {len(val_triples)} examples")
    print("  eval mode:  closed-domain per-example (ranking within each question's support set)")

    all_results: dict[str, dict] = {}

    # ---- BM25 ---------------------------------------------------------------
    print("\n[1] BM25 …")
    all_results["bm25"] = evaluate_retriever("bm25", val_triples, corpus, k_values)

    if not args.baselines_only:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device: {device}")

        from train import Encoder

        def _run_dense(label: str, q_model: str, a_model: str) -> dict:
            print(f"\n  Loading {label} encoders …")
            q_enc = Encoder(q_model).to(device)
            a_enc = Encoder(a_model).to(device)
            print(f"  Pre-encoding {len(corpus)} passages …")
            embs = _precompute_embs(corpus, a_enc, device)
            return evaluate_retriever(
                "dense", val_triples, corpus, k_values,
                query_encoder=q_enc, precomputed_embs=embs, device=device,
            )

        # ---- Vanilla MedCPT -------------------------------------------------
        print("\n[2] Vanilla MedCPT (zero-shot) …")
        all_results["medcpt_vanilla"] = _run_dense(
            "MedCPT",
            "ncbi/MedCPT-Query-Encoder",
            "ncbi/MedCPT-Article-Encoder",
        )

        # ---- Fine-tuned MedCPT ----------------------------------------------
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                print(f"  WARNING: checkpoint not found: {ckpt_path}", file=sys.stderr)
            else:
                print(f"\n[3] Fine-tuned MedCPT ({ckpt_path.name}) …")
                all_results["medcpt_finetuned"] = _run_dense(
                    "fine-tuned MedCPT",
                    str(ckpt_path / "query_encoder"),
                    str(ckpt_path / "article_encoder"),
                )
        else:
            print("\n[3] Fine-tuned MedCPT — skipped (pass --checkpoint to enable)")

        # ---- SBERT ----------------------------------------------------------
        print("\n[4] SBERT …")
        all_results["sbert"] = _run_dense(
            "SBERT",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        # ---- DPR ------------------------------------------------------------
        print("\n[5] DPR …")
        all_results["dpr"] = _run_dense(
            "DPR",
            "facebook/dpr-question_encoder-single-nq-base",
            "facebook/dpr-ctx_encoder-single-nq-base",
        )

    # ---- print + save -------------------------------------------------------
    print_table(all_results, k_values)

    existing: dict = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    existing["retriever"] = all_results
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved → {RESULTS_FILE}")

    save_txt_report(all_results, k_values, RESULTS_TXT, args.checkpoint)


if __name__ == "__main__":
    main()
