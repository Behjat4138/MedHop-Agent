"""
Public retriever interface for the agentic RAG pipeline.

Exposes:
  retrieve(question, corpus=None, k=5) -> list[str]

Loading priority (lazy, on first call):
  1. Fine-tuned MedCPT  — uses the latest checkpoint in checkpoints/
  2. BM25               — fallback if no checkpoint exists or loading fails

Ad-hoc corpus: if the caller passes a list of passage strings, a temporary
BM25 index is built over them (useful for closed-domain, per-question retrieval).

Example:
  from retriever.retriever import retrieve
  passages = retrieve("interacts_with DB00773?", k=5)
"""

import json
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).parent / "data"
_CKPT_DIR = Path(__file__).parent / "checkpoints"

# Lazy-loaded singletons
_retriever = None
_corpus_ids: list[str] | None = None
_corpus_texts: list[str] | None = None


def _latest_checkpoint() -> Optional[Path]:
    if not _CKPT_DIR.exists():
        return None
    ckpts = sorted(_CKPT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
    return ckpts[-1] if ckpts else None


def _load_default_retriever() -> None:
    """Populate the module-level singletons on first use."""
    global _retriever, _corpus_ids, _corpus_texts

    corpus_path = _DATA_DIR / "corpus.json"
    if not corpus_path.exists():
        print(
            "[retriever] WARNING: corpus.json not found — run data_prep.py first.\n"
            "             Returning empty results until the corpus is built."
        )
        return

    with open(corpus_path) as f:
        corpus: dict[str, str] = json.load(f)
    _corpus_ids = list(corpus.keys())
    _corpus_texts = [corpus[pid] for pid in _corpus_ids]

    ckpt = _latest_checkpoint()
    if ckpt:
        try:
            from baselines import make_medcpt
            _retriever = make_medcpt(_corpus_ids, _corpus_texts, checkpoint_dir=str(ckpt))
            print(f"[retriever] Loaded fine-tuned MedCPT from {ckpt.name}")
            return
        except Exception as exc:
            print(f"[retriever] Fine-tuned MedCPT failed ({exc}); falling back to BM25")

    from baselines import BM25Retriever
    _retriever = BM25Retriever(_corpus_ids, _corpus_texts)
    print("[retriever] Loaded BM25 retriever")


def retrieve(
    question: str,
    corpus: Optional[list[str]] = None,
    k: int = 5,
) -> list[str]:
    """
    Retrieve top-k relevant passages for a biomedical question.

    Args:
        question: Input query string.
        corpus:   Optional list of passage strings to search over.
                  When provided, a temporary BM25 index is built on-the-fly.
                  When None, the pre-built MedHop corpus is used.
        k:        Number of passages to return.

    Returns:
        List of up to k passage strings, ranked by relevance.
    """
    global _retriever

    if corpus is not None:
        # Closed-domain path: build a throw-away BM25 index over the supplied passages
        from baselines import BM25Retriever
        tmp_ids = [str(i) for i in range(len(corpus))]
        tmp = BM25Retriever(tmp_ids, corpus)
        return [corpus[int(pid)] for pid, _ in tmp.retrieve(question, k=k)]

    # Open-domain path: use the pre-built global retriever
    if _retriever is None:
        _load_default_retriever()
    if _retriever is None:
        return []

    pid_to_text = dict(zip(_corpus_ids or [], _corpus_texts or []))
    return [pid_to_text.get(pid, "") for pid, _ in _retriever.retrieve(question, k=k)]
