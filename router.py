"""
Query router — public interface for the RAG pipeline.

Teammates import this module and call predict_hop().

    from router import predict_hop
    decision = predict_hop(question, candidates, supports)
    # returns 'single' or 'multi'

Note on deployment: features require the support passages bundled with MedHop.
In a live pipeline where passages aren't available pre-retrieval, the router
would fall back to predict_multi() or use only n_candidates as a proxy.
"""

import pickle
import numpy as np

from features import extract_features


# Load the trained model once at import time
_MODEL_PATH = "model.pkl"
try:
    with open(_MODEL_PATH, "rb") as _f:
        _clf = pickle.load(_f)
    _model_loaded = True
except FileNotFoundError:
    _clf = None
    _model_loaded = False


def predict_hop(question: str, candidates: list[str], supports: list[str]) -> str:
    """
    Predict whether a question needs single-hop or multi-hop retrieval.

    Args:
        question:   The query string (e.g. "interacts_with DB00773?")
        candidates: List of candidate answer strings
        supports:   List of support passage strings

    Returns:
        'single' if one retrieval step is likely sufficient, else 'multi'
    """
    if not _model_loaded:
        # Fallback if model file is missing — always predict multi (safer default)
        return "multi"

    example = {"query": question, "candidates": candidates, "supports": supports}
    vec = extract_features(example).reshape(1, -1)
    label = int(_clf.predict(vec)[0])
    return "single" if label == 0 else "multi"


def predict_multi() -> str:
    """Baseline: always predict multi-hop. Used for ablation comparison."""
    return "multi"


if __name__ == "__main__":
    # Smoke test with a fake example
    fake_question   = "interacts_with DB00773?"
    fake_candidates = ["DB00072", "DB00294", "DB00338"]
    fake_supports   = [
        "DB00773 has been shown to interact with DB00072 in clinical trials.",
        "Studies show DB00294 affects the same pathway as DB00338.",
    ]

    if _model_loaded:
        result = predict_hop(fake_question, fake_candidates, fake_supports)
        print(f"predict_hop() → '{result}'")
        print("router.py is working correctly.")
    else:
        print("model.pkl not found — run train.py first.")
