"""
Cross-encoder reranker.

Unlike a bi-encoder (Person A's retriever), which embeds query and passage
independently and scores with cosine similarity, a cross-encoder concatenates
them into a single [CLS] query [SEP] passage [SEP] input and lets every token
attend to every other.  More accurate per pair, but slower — which is fine
here because the reranker only sees the top 20-30 passages from the retriever,
not the full corpus.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, CPU-feasible
  - Pretrained on MS-MARCO passage ranking (pairwise ranking objective)
  - Used zero-shot — fine-tuning on MedHop is future work
"""
import numpy as np
from sentence_transformers import CrossEncoder

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_length: int = 512,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One score per (query, passage) pair."""
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Returns [(passage, score), ...] sorted best-first.
        If top_k is None, returns all passages ranked.
        """
        scores = self.score(query, passages)
        idx = np.argsort(scores)[::-1]
        if top_k is not None:
            idx = idx[:top_k]
        return [(passages[i], scores[i]) for i in idx]


# ---------------------------------------------------------------------------
# Convenience singleton for Person D's agent loop
# ---------------------------------------------------------------------------

_default: Reranker | None = None


def rerank(query: str, passages: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    """
    Lazy-loaded convenience wrapper.  First call loads the model; subsequent
    calls reuse it.  Suitable for Person D's pipeline glue.

    Example:
        from reranker import rerank
        top_passages = rerank(query, retriever_output, top_k=5)
    """
    global _default
    if _default is None:
        _default = Reranker()
    return _default.rerank(query, passages, top_k=top_k)


if __name__ == "__main__":
    # Smoke test
    r = Reranker()
    query = "interacts_with DB00773?"
    passages = [
        "DB00773 has been shown to interact with DB00072 in clinical trials.",
        "Unrelated passage about weather.",
        "Studies show DB00294 affects the same pathway as DB00338.",
    ]
    ranked = r.rerank(query, passages, top_k=3)
    print("Ranked passages (best first):")
    for i, (p, s) in enumerate(ranked, 1):
        print(f"  {i}. score={s:+.3f}  {p[:70]}")