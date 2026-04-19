# Query Router — CS 572 Final Project

## Project Role
This repo owns **only the router component** of a team-built agentic RAG pipeline for biomedical QA.

Pipeline (full team): `question → router (this repo) → retriever → reranker → generator → answer`

The router's job: given a biomedical question, predict whether it needs **1-hop** or **multi-hop** retrieval.

## Dataset
**MedHop** from QAngaroo benchmark (Welbl et al. 2018). Loaded via HuggingFace `datasets`.
- Do NOT use MedQA or MedHopQA — different benchmarks.
- MedHop does not provide ground-truth hop-count labels. Labels must be derived via weak supervision (BM25).

## Deliverables (priority order — drop from bottom if time runs out)
1. Load MedHop + compute basic statistics (question count, passages per question, candidate count distribution)
2. Feature extraction function: `extract_features(question: str) -> np.ndarray` — 3–5 features max
3. Weak-supervision labeling via BM25: label each training question as `easy` (1-hop) or `hard` (multi-hop)
4. Classifier: scikit-learn `LogisticRegression` or `RandomForestClassifier` trained on BM25-derived labels
5. Evaluation: accuracy on held-out set + feature importance analysis, logged to `results.json`
6. (stretch) Mini ablation: router vs. always-predict-multi-hop baseline

## Interface Contract (for teammate handoff)
The router must expose this function signature so the retriever can call it:
```python
def predict_hop(question: str) -> str:
    """Returns 'single' or 'multi'."""
```

## Tech Stack
- Python 3.10+
- `datasets` (HuggingFace) — MedHop loading
- `rank_bm25` — BM25 labeling
- `scikit-learn` — classifier
- `scispacy` + `en_core_sci_sm` — only if NER features are used (optional)
- `numpy`, `matplotlib` — features and plots
- Results logged to `results.json` (no W&B)

## Constraints
- No deep learning. Scikit-learn only.
- Favor clear code over clever code. Comments explain *why*, not *what*.
- Flat repo structure — no nested src/ packages.
- Pin all dependencies in `requirements.txt`.

## Owner
Student: Qian Jingyun (jingyun.qian@emory.edu)
Course: CS 572, Spring 2026
