# Query Router

**Owner:** Qian Jingyun (jingyun.qian@emory.edu) — CS 572, Spring 2026

## Project Summary

The query router is a lightweight classifier that sits at the front of the RAG pipeline and predicts whether a biomedical question requires single-hop or multi-hop retrieval. It is trained on the MedHop dataset (Welbl et al. 2018) using weak supervision: since MedHop provides no ground-truth hop-count labels, we derive proxy labels by scoring how well BM25 can surface the answer drug from the support passages in one step, then splitting at the training-set median to produce a 50/50 binary label. A logistic regression classifier with five structural features (passage count, candidate count, BM25 scores) achieves 69.3% accuracy on the validation split, compared to 62.6% for an always-multi baseline — a modest but consistent improvement that lets the pipeline skip the more expensive multi-hop retrieval path on questions where a single step is likely sufficient.

---

## Pipeline Context

```
question → [router] → retriever → reranker → generator → answer
```

The router's output tells the retriever how many retrieval steps to run. Routing is purely a cost-efficiency decision: single-hop is faster and cheaper; multi-hop finds more evidence but takes more time and compute.

---

## Quickstart for Teammates

### 1. Install dependencies

```bash
conda create -n cs572 python=3.10 -y
conda activate cs572
pip install -r requirements.txt
```

### 2. Download the pre-trained model

`model.pkl` is committed to this repo. No training needed unless you want to retrain.

### 3. Call the router from your code

```python
from router.router import predict_hop

decision = predict_hop(
    question="interacts_with DB00773?",
    candidates=["DB00072", "DB00294", "DB00338"],
    supports=[
        "DB00773 has been shown to interact with DB00072 in clinical trials.",
        "Studies show DB00294 affects the same pathway as DB00338.",
    ],
)
# decision is 'single' or 'multi'
```

**Important:** the router needs the support passages to extract features. In a live pipeline where passages are not yet retrieved, fall back to `'multi'` (the safer default) or pass an empty list — the router will return `'multi'` automatically if the model file is missing.

---

## File Overview

| File | Purpose |
|---|---|
| `router.py` | **Public interface** — import this. Exposes `predict_hop()`. |
| `features.py` | Feature extraction from support passages. |
| `label.py` | BM25 weak-supervision labeling (generates `labels_train.json`). |
| `train.py` | Train the classifier, saves `model.pkl`. |
| `evaluate.py` | Evaluate on MedHop validation split, saves `results.json` and plots. |
| `explore.py` | Dataset statistics (one-time exploration). |
| `model.pkl` | Trained logistic regression pipeline (committed). |
| `labels_train.json` | BM25-derived binary labels for training set (committed). |
| `results.json` | Validation accuracy, baselines, feature importances. |
| `results/` | Confusion matrix and feature importance plots. |

---

## Retraining from Scratch

Only needed if you want to re-derive labels or change the classifier.

```bash
cd router/

# Step 1: derive weak-supervision labels from training set (~2 min)
conda run -n cs572 python label.py

# Step 2: train classifier and run 5-fold CV
conda run -n cs572 python train.py

# Step 3: evaluate on validation split
conda run -n cs572 python evaluate.py
```

---

## Results

| Method | Validation Accuracy |
|---|---|
| Logistic Regression classifier | **69.3%** |
| Always-multi baseline | 62.6% |
| Always-single baseline | 37.4% |

Top features by coefficient magnitude:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `n_supports` | −0.89 | More passages → easier to answer in one step |
| `top_bm25_score` | +0.14 | Higher BM25 score → harder (answer needs more searching) |

---

## Known Limitations

- **Labels are proxies, not ground truth.** MedHop has no hop-count labels. Labels reflect relative BM25 retrieval difficulty, not true 1-hop vs. multi-hop distinctions.
- **Features require pre-fetched passages.** In a live pipeline without passages available upfront, only `n_candidates` is computable. The router falls back to `'multi'` if `model.pkl` is missing.
- **MedHop is entirely multi-hop by design.** The "single" label means "easier multi-hop," not "genuinely 1-hop." End-to-end validation against retrieval accuracy is the correct test, which requires the full pipeline.
