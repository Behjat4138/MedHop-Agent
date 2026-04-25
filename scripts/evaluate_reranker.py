"""
Evaluate the cross-encoder reranker on MedHop validation.

Protocol: identical to Person A's retriever/evaluate.py (closed-domain).
  For each validation example:
    - Take its ~30 support passages
    - Rank them with the reranker
    - Relevant = passages that contain the answer drug-ID as substring
    - Report Recall@K, nDCG@K, MAP, MRR

Numbers are directly comparable to entries under results.json -> "retriever".

Run from repo root:
    python scripts/evaluate_reranker.py
    python scripts/evaluate_reranker.py --run_name ms_marco_zero_shot
    python scripts/evaluate_reranker.py --compare_to_retriever
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Make imports work when running from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reranker.reranker import Reranker
from eval_harness.runner import evaluate_rankings, append_to_leaderboard


DATA_DIR = REPO_ROOT / "retriever" / "data"
RESULTS_FILE = REPO_ROOT / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        default="ms_marco_zero_shot",
        help="Key to store results under in results.json -> 'reranker'",
    )
    parser.add_argument(
        "--model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace cross-encoder model name",
    )
    parser.add_argument(
        "--compare_to_retriever",
        action="store_true",
        help="Print a comparison table vs existing retriever results",
    )
    args = parser.parse_args()

    # ---- load data --------------------------------------------------------
    print("Loading data ...")
    if not (DATA_DIR / "corpus.json").exists():
        print(f"ERROR: {DATA_DIR}/corpus.json not found.")
        print("Run Person A's data_prep.py first (from inside retriever/):")
        print("    cd retriever && python data_prep.py")
        sys.exit(1)

    with open(DATA_DIR / "corpus.json") as f:
        corpus: dict[str, str] = json.load(f)
    with open(DATA_DIR / "val_triples.json") as f:
        val_triples: list[dict] = json.load(f)

    print(f"  corpus:  {len(corpus)} passages")
    print(f"  val set: {len(val_triples)} examples")
    print(f"  eval:    closed-domain (rank each question's own support set)")

    # ---- load reranker ----------------------------------------------------
    print(f"\nLoading reranker: {args.model}")
    reranker = Reranker(model_name=args.model)

    # ---- rerank each example ----------------------------------------------
    rankings: dict[str, list[str]] = {}
    relevants: dict[str, set[str]] = {}
    skipped_no_support = 0
    skipped_no_answer = 0

    for triple in tqdm(val_triples, desc="Reranking"):
        # Keep only pids we have text for
        support_pids = [p for p in triple["supports"] if p in corpus]
        if not support_pids:
            skipped_no_support += 1
            continue

        # Relevant = passages that contain the answer drug-ID (matches Person A)
        relevant = {pid for pid in support_pids if triple["answer"] in corpus[pid]}
        if not relevant:
            # No signal on this example — skip rather than pollute the mean
            skipped_no_answer += 1
            continue

        # Score all passages, then sort pids (not text) to avoid duplicate-text ambiguity
        passages = [corpus[pid] for pid in support_pids]
        scores = reranker.score(triple["query"], passages)
        ranked_pairs = sorted(
            zip(support_pids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        rankings[triple["id"]] = [pid for pid, _ in ranked_pairs]
        relevants[triple["id"]] = relevant

    print(f"\nEvaluated {len(rankings)} examples")
    print(f"  skipped {skipped_no_support} with no support passages")
    print(f"  skipped {skipped_no_answer} with no answer-containing support passage")

    # ---- metrics ----------------------------------------------------------
    metrics = evaluate_rankings(rankings, relevants, k_values=(1, 5, 10))

    print(f"\n=== Reranker: {args.run_name} ===")
    for k in ("recall@1", "recall@5", "recall@10", "ndcg@1", "ndcg@5", "ndcg@10", "map", "mrr"):
        if k in metrics:
            print(f"  {k:<12} {metrics[k]:.4f}")

    # ---- persist ----------------------------------------------------------
    append_to_leaderboard("reranker", args.run_name, metrics, RESULTS_FILE)
    print(f"\nAppended to {RESULTS_FILE} under 'reranker' > '{args.run_name}'")

    # ---- optional comparison ---------------------------------------------
    if args.compare_to_retriever:
        print("\n=== Comparison vs retriever baselines ===")
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)

        if "retriever" not in all_results:
            print("  (no retriever results in results.json yet)")
            return

        metric_keys = ("recall@1", "recall@5", "recall@10",
                       "ndcg@1", "ndcg@5", "ndcg@10")
        header = f"{'Model':<28}" + "".join(f"{k:>10}" for k in metric_keys)
        print(header)
        print("-" * len(header))

        for name, r in all_results["retriever"].items():
            row = f"{name:<28}"
            row += "".join(f"{r.get(k, 0):>10.4f}" for k in metric_keys)
            print(row)

        row = f"{args.run_name + ' (reranker)':<28}"
        row += "".join(f"{metrics.get(k, 0):>10.4f}" for k in metric_keys)
        print(row)


if __name__ == "__main__":
    main()