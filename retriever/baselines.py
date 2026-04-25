"""
Retrieval baselines for comparison with fine-tuned MedCPT.

All four retrievers expose the same interface:
  .retrieve(query: str, k: int = 5) -> list[tuple[str, float]]
  Returns [(passage_id, score), …] sorted highest-score first.

Retrievers:
  BM25Retriever         — lexical (rank_bm25)
  BiEncoderRetriever    — general dense bi-encoder (used for MedCPT, SBERT, DPR)

Factory helpers:
  make_medcpt(corpus_ids, corpus_texts, checkpoint_dir=None)
  make_sbert(corpus_ids, corpus_texts)
  make_dpr(corpus_ids, corpus_texts)
"""

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from torch.nn.functional import normalize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, corpus_ids: list[str], corpus_texts: list[str]):
        self.ids = corpus_ids
        self.texts = corpus_texts
        tokenized = [t.lower().split() for t in tqdm(corpus_texts, desc="BM25 indexing")]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_k = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top_k]


# ---------------------------------------------------------------------------
# Dense bi-encoder (shared by MedCPT, SBERT, DPR)
# ---------------------------------------------------------------------------

class BiEncoderRetriever:
    """
    Asymmetric bi-encoder: separate query and article encoders.
    For symmetric models (SBERT) pass the same model name twice.

    Call .index(corpus_ids, corpus_texts) once to build the embedding index,
    then call .retrieve(query, k) for each query.
    """

    def __init__(
        self,
        query_model: str,
        article_model: str | None = None,
        max_query_len: int = 64,
        max_article_len: int = 512,
        device: str | None = None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_q = max_query_len
        self.max_a = max_article_len

        self.q_tok = AutoTokenizer.from_pretrained(query_model)
        self.q_model = AutoModel.from_pretrained(query_model).to(self.device)

        art = article_model or query_model
        if art == query_model:
            self.a_tok = self.q_tok
            self.a_model = self.q_model
        else:
            self.a_tok = AutoTokenizer.from_pretrained(art)
            self.a_model = AutoModel.from_pretrained(art).to(self.device)

        self._corpus_ids: list[str] | None = None
        self._corpus_embs: torch.Tensor | None = None

    def _encode(
        self,
        texts: list[str],
        tokenizer,
        model,
        max_len: int,
        batch_size: int = 32,
        desc: str = "",
    ) -> torch.Tensor:
        all_embs: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
                enc = tokenizer(
                    texts[i : i + batch_size],
                    max_length=max_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                out = model(**enc)
                # DPR models expose pooler_output; BERT/MedCPT/SBERT expose last_hidden_state
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    cls = out.pooler_output
                else:
                    cls = out.last_hidden_state[:, 0, :]
                all_embs.append(normalize(cls, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)

    def index(
        self, corpus_ids: list[str], corpus_texts: list[str], batch_size: int = 32
    ) -> None:
        self._corpus_ids = corpus_ids
        self._corpus_embs = self._encode(
            corpus_texts, self.a_tok, self.a_model, self.max_a, batch_size, "Indexing"
        )

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if self._corpus_embs is None:
            raise RuntimeError("Call .index() before .retrieve()")
        q_emb = self._encode([query], self.q_tok, self.q_model, self.max_q)  # [1, D]
        scores = (q_emb @ self._corpus_embs.T).squeeze(0).numpy()
        top_k = np.argsort(scores)[::-1][:k]
        return [(self._corpus_ids[i], float(scores[i])) for i in top_k]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_medcpt(
    corpus_ids: list[str],
    corpus_texts: list[str],
    checkpoint_dir: str | None = None,
) -> BiEncoderRetriever:
    """
    Zero-shot MedCPT if checkpoint_dir is None;
    fine-tuned MedCPT if a checkpoint path is provided.
    """
    from pathlib import Path

    q_model = "ncbi/MedCPT-Query-Encoder"
    a_model = "ncbi/MedCPT-Article-Encoder"
    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        q_model = str(ckpt / "query_encoder")
        a_model = str(ckpt / "article_encoder")

    ret = BiEncoderRetriever(q_model, a_model)
    ret.index(corpus_ids, corpus_texts)
    return ret


def make_sbert(
    corpus_ids: list[str],
    corpus_texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> BiEncoderRetriever:
    """SBERT symmetric retriever (query and passage use the same encoder)."""
    ret = BiEncoderRetriever(model_name, model_name, max_query_len=256, max_article_len=256)
    ret.index(corpus_ids, corpus_texts)
    return ret


def make_dpr(
    corpus_ids: list[str],
    corpus_texts: list[str],
) -> BiEncoderRetriever:
    """DPR with Wikipedia-trained asymmetric encoders (open-domain baseline)."""
    ret = BiEncoderRetriever(
        "facebook/dpr-question_encoder-single-nq-base",
        "facebook/dpr-ctx_encoder-single-nq-base",
    )
    ret.index(corpus_ids, corpus_texts)
    return ret
