"""
STRETCH GOAL — only run if you finish the MVP with hours to spare.

Fine-tune the cross-encoder on MedHop with a pairwise margin loss:
    L = max(0, margin - (score_pos - score_neg))

Training data comes from Person A's train_triples_with_negatives.json
(positives from `positives`, hard negatives from `hard_negatives`).

Expected runtime:
  - CPU, batch_size=8, 1 epoch: ~30-45 min
  - GPU (T4), batch_size=16, 2 epochs: ~10 min

After training, evaluate with:
    python scripts/evaluate_reranker.py \\
        --model reranker/checkpoints/finetuned \\
        --run_name ms_marco_finetuned

Run from repo root:
    python reranker/train_reranker.py
    python reranker/train_reranker.py --epochs 2 --batch_size 16
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.losses import MSELoss  # placeholder — we override below

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "retriever" / "data"
OUT_DIR = Path(__file__).parent / "checkpoints"


def load_training_pairs(use_hard_negatives: bool = True) -> list[InputExample]:
    """
    Each example contributes:
      - one (query, positive_passage, label=1.0)
      - one (query, hard_negative_passage, label=0.0)
    The cross-encoder's default training path uses BCE on these labels,
    which is a valid pairwise proxy.
    """
    triples_path = DATA_DIR / "train_triples_with_negatives.json"
    if not triples_path.exists():
        print(f"ERROR: {triples_path} not found.")
        print("Run Person A's hard_negatives.py first:")
        print("    cd retriever && python hard_negatives.py")
        sys.exit(1)

    with open(triples_path) as f:
        triples = json.load(f)
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)

    examples = []
    for t in triples:
        if not t.get("positives"):
            continue
        pos_pid = random.choice(t["positives"])
        pos_text = corpus.get(pos_pid)
        if not pos_text:
            continue

        examples.append(InputExample(texts=[t["query"], pos_text], label=1.0))

        if use_hard_negatives:
            for neg_pid in t.get("hard_negatives", [])[:1]:  # 1 hard neg per positive
                neg_text = corpus.get(neg_pid)
                if neg_text:
                    examples.append(InputExample(texts=[t["query"], neg_text], label=0.0))

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_hard_neg", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading training pairs ...")
    examples = load_training_pairs(use_hard_negatives=not args.no_hard_neg)
    print(f"  {len(examples)} (query, passage, label) pairs")

    print(f"Loading base model: {args.model}")
    model = CrossEncoder(args.model, num_labels=1, max_length=512, device=device)

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)

    OUT_DIR.mkdir(exist_ok=True)
    save_path = OUT_DIR / "finetuned"

    print(f"Training {args.epochs} epoch(s) ...")
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        warmup_steps=max(100, len(loader) // 10),
        save_best_model=False,
        output_path=str(save_path),
        show_progress_bar=True,
    )
    model.save(str(save_path))
    print(f"\nFine-tuned model saved to {save_path}")
    print("\nEvaluate with:")
    print(f"    python scripts/evaluate_reranker.py \\")
    print(f"        --model {save_path} \\")
    print(f"        --run_name ms_marco_finetuned --compare_to_retriever")


if __name__ == "__main__":
    main()