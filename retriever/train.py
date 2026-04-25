"""
Fine-tune MedCPT on MedHop using contrastive (InfoNCE) learning.

Architecture: asymmetric bi-encoder
  Query encoder   — ncbi/MedCPT-Query-Encoder   (max 64 tokens)
  Article encoder — ncbi/MedCPT-Article-Encoder  (max 512 tokens)

Loss: InfoNCE with in-batch negatives + optional BM25 hard negatives.
  In-batch: every positive from another query in the same batch is a negative.
  Hard neg:  pre-mined passages (from hard_negatives.py) added to the passage pool.

Checkpoints saved after every epoch to checkpoints/epoch_N/.

Usage:
  python train.py
  python train.py --epochs 10 --batch_size 32 --lr 1e-5
  python train.py --no_hard_neg   # in-batch negatives only
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Wraps a HuggingFace BERT-style model and produces unit-normalized
    [CLS]-token embeddings — the convention used by MedCPT.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # DPR models expose pooler_output; BERT/MedCPT expose last_hidden_state
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            cls = out.pooler_output
        else:
            cls = out.last_hidden_state[:, 0, :]
        return normalize(cls, dim=-1)

    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Encode a list of strings to a [N, D] embedding tensor (no grad)."""
        self.eval()
        all_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size],
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                all_embs.append(self(**enc).cpu())
        return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """
    Each item: (query, positive_passage, [hard_negative_passages]).
    Samples one positive uniformly at random from the triple's positive list.
    """

    def __init__(
        self,
        triples: list[dict],
        corpus: dict[str, str],
        use_hard_negatives: bool = True,
        n_hard_neg: int = 1,
    ):
        self.items: list[dict] = []
        missing = 0
        for t in triples:
            pos_pid = random.choice(t["positives"]) if t["positives"] else None
            pos_text = corpus.get(pos_pid or "", "")
            if not pos_text:
                missing += 1
                continue

            neg_texts: list[str] = []
            if use_hard_negatives:
                neg_pids = t.get("hard_negatives", [])[:n_hard_neg]
                neg_texts = [corpus[p] for p in neg_pids if p in corpus]

            self.items.append(
                {"query": t["query"], "positive": pos_text, "hard_negatives": neg_texts}
            )
        if missing:
            print(f"  [dataset] dropped {missing} examples with no positive passage text")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _collate(batch: list[dict]):
    queries = [x["query"] for x in batch]
    positives = [x["positive"] for x in batch]
    hard_negs: list[str] = []
    for x in batch:
        hard_negs.extend(x["hard_negatives"])
    return queries, positives, hard_negs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def infonce_loss(
    q_embs: torch.Tensor,
    p_embs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard InfoNCE with in-batch negatives.
    q_embs: [B, D], p_embs: [B+, D]  (positives are on the diagonal).
    """
    sim = q_embs @ p_embs.T / temperature  # [B, B+]
    labels = torch.arange(len(q_embs), device=q_embs.device)
    return nn.CrossEntropyLoss()(sim, labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    q_enc: Encoder,
    a_enc: Encoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float,
) -> float:
    q_enc.train()
    a_enc.train()
    total_loss = 0.0

    for queries, positives, hard_negs in tqdm(loader, desc="  batches", leave=False):
        optimizer.zero_grad()

        q_enc_input = q_enc.tokenizer(
            queries, max_length=64, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        p_enc_input = a_enc.tokenizer(
            positives, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        q_embs = q_enc(**q_enc_input)   # [B, D]
        p_embs = a_enc(**p_enc_input)   # [B, D]

        loss = infonce_loss(q_embs, p_embs, temperature)

        # Augment passage pool with hard negatives when present
        if hard_negs:
            hn_input = a_enc.tokenizer(
                hard_negs, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            hn_embs = a_enc(**hn_input)  # [H, D]
            aug_p_embs = torch.cat([p_embs, hn_embs], dim=0)  # [B+H, D]
            hn_loss = infonce_loss(q_embs, aug_p_embs, temperature)
            loss = (loss + hn_loss) / 2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(q_enc.parameters()) + list(a_enc.parameters()), max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _save_checkpoint(q_enc: Encoder, a_enc: Encoder, epoch: int) -> Path:
    ckpt = CHECKPOINT_DIR / f"epoch_{epoch}"
    q_enc.model.save_pretrained(ckpt / "query_encoder")
    q_enc.tokenizer.save_pretrained(ckpt / "query_encoder")
    a_enc.model.save_pretrained(ckpt / "article_encoder")
    a_enc.tokenizer.save_pretrained(ckpt / "article_encoder")
    return ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--n_hard_neg", type=int, default=1)
    parser.add_argument("--no_hard_neg", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Prefer triples with pre-mined hard negatives
    triples_path = DATA_DIR / "train_triples_with_negatives.json"
    if not triples_path.exists():
        triples_path = DATA_DIR / "train_triples.json"
        print("Note: hard-negative file not found — run hard_negatives.py first for best results")

    print("Loading data …")
    with open(triples_path) as f:
        triples = json.load(f)
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)

    dataset = ContrastiveDataset(
        triples,
        corpus,
        use_hard_negatives=not args.no_hard_neg,
        n_hard_neg=args.n_hard_neg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,  # tokenizers are not fork-safe
    )
    print(f"Training set: {len(dataset)} examples  |  {len(loader)} batches/epoch")

    print("Loading MedCPT encoders …")
    q_enc = Encoder(_QUERY_MODEL).to(device)
    a_enc = Encoder(_ARTICLE_MODEL).to(device)

    optimizer = torch.optim.AdamW(
        list(q_enc.parameters()) + list(a_enc.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    # Linear warmup + cosine decay
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(q_enc, a_enc, loader, optimizer, device, args.temperature)
        scheduler.step()
        ckpt = _save_checkpoint(q_enc, a_enc, epoch)
        print(f"Epoch {epoch}/{args.epochs}  loss={loss:.4f}  → {ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    main()
