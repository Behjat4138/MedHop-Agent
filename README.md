# MedHop-Agent

**Efficient Agentic Multi-Hop Retrieval for Biomedical Question Answering**

CS 572 Final Project · Spring 2026 · Emory University

---

## Overview

MedHop-Agent is a retrieval-augmented question answering system designed for biomedical queries that require reasoning across multiple documents. Standard RAG pipelines retrieve once and generate, which fails when later evidence depends on earlier partial inferences — a common pattern in drug-interaction and clinical reasoning questions. This project addresses that gap with a four-stage modular pipeline:

```
Question → Router → Retriever → Reranker → Agent → Answer
```

Each stage is independently trainable and replaceable. The system is evaluated on the **MedHop** benchmark, which tests multi-hop reasoning across drug-interaction passages.

---
## Headline result:
The fine-tuned cross-encoder reranker achieves Recall@5 = 0.567 and Recall@10 = 0.868 on the MedHop validation split, improving over the strongest retriever-only baseline (fine-tuned MedCPT) by +15.5 and +17.5 percentage points respectively.
## Pipeline Components

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | **Router** | Classifies whether a question requires single-hop or multi-hop retrieval |
| 2 | **Retriever** | Fine-tuned MedCPT bi-encoder; retrieves top-30 candidate passages |
| 3 | **Reranker** | Cross-encoder; re-scores the top-30 to produce the top-5 |
| 4 | **Agent** | Multi-hop retrieve-reason loop with early stopping; generates final answer |

The router lets the system skip the expensive multi-hop path on easy questions, the retriever provides broad recall, the reranker provides precision, and the agent loop handles iterative reasoning when needed.

---

## Repository Structure

```
cs572-finalProject/
├── retriever/              # Stage 2 — fine-tuned MedCPT (Person A)
│   ├── data_prep.py        # Build MedHop triples from HuggingFace
│   ├── baselines.py        # BM25, vanilla MedCPT, SBERT, DPR
│   ├── train.py            # Contrastive fine-tuning (InfoNCE + hard negatives)
│   ├── hard_negatives.py   # BM25 + dense hard negative mining
│   ├── evaluate.py         # Closed-domain Recall@K / nDCG@K
│   └── retriever.py        # Public retrieve() interface
├── router/                 # Stage 1 — query router (Person C)
│   ├── features.py         # Structural features (BM25 scores, passage counts)
│   ├── label.py            # Weak-supervision labels via BM25 median split
│   ├── train.py            # Logistic regression + 5-fold CV
│   └── router.py           # Public predict_hop() interface
├── reranker/               # Stage 3 — cross-encoder reranker (Person B)
│   ├── reranker.py         # Public Reranker / rerank() interface
│   └── train_reranker.py   # Fine-tuning with pairwise margin loss
├── eval_harness/           # Shared evaluation framework (Person B)
│   ├── metrics.py          # Recall@K, nDCG@K, MAP, MRR, EM
│   └── runner.py           # Aggregation + leaderboard merge
├── scripts/
│   └── evaluate_reranker.py
├── results.json            # Shared leaderboard across all components
└── README.md
```

---

## Results on MedHop Validation (342 examples, closed-domain)

Closed-domain evaluation: each question's ~30 curated support passages are reranked among themselves. Relevant set = passages containing the answer drug-ID as a substring.

### Retrieval Stage

| Model | Recall@5 | nDCG@5 | Recall@10 | nDCG@10 |
|-------|---------:|-------:|----------:|--------:|
| BM25 | 0.199 | 0.110 | 0.409 | 0.176 |
| DPR | 0.170 | 0.089 | 0.386 | 0.154 |
| Vanilla MedCPT | 0.246 | 0.137 | 0.491 | 0.215 |
| SBERT | 0.342 | 0.188 | 0.556 | 0.256 |
| **Fine-tuned MedCPT** | **0.412** | **0.230** | **0.693** | **0.316** |

### Reranking Stage

| Model | Recall@5 | nDCG@5 | Recall@10 | nDCG@10 |
|-------|---------:|-------:|----------:|--------:|
| Zero-shot ms-marco cross-encoder | 0.336 | 0.172 | 0.585 | 0.248 |
| **Fine-tuned cross-encoder (MedHop)** | **0.567** | **0.343** | **0.868** | **0.446** |

### Routing Stage

| Model | Accuracy | Δ over best baseline |
|-------|---------:|---------------------:|
| Always-multi (baseline) | 0.626 | — |
| Always-single (baseline) | 0.374 | — |
| **Logistic Regression (5 features)** | **0.693** | **+6.7 pp** |

---

## Quickstart

### Prerequisites

- Python 3.10+
- ~5GB disk for models and HuggingFace dataset cache
- GPU optional but recommended for retriever / reranker fine-tuning

### Installation

```bash
git clone <repo-url>
cd cs572-finalProject

# Create environment
conda create -n cs572 python=3.10 -y
conda activate cs572

# Install dependencies for each component
pip install -r retriever/requirements.txt
pip install -r reranker/requirements.txt
pip install -r router/requirements.txt
```

### Reproducing the Results

**Retriever** (Person A's component — bi-encoder fine-tuning):

```bash
cd retriever
python data_prep.py            # Build MedHop triples (~3 min)
python hard_negatives.py       # Mine BM25 hard negatives (~5 min)
python train.py --epochs 5     # Fine-tune MedCPT (~30 min on GPU)
python evaluate.py --checkpoint checkpoints/epoch_5
cd ..
```

**Reranker** (Person B's component — cross-encoder):

```bash
# Zero-shot evaluation
python scripts/evaluate_reranker.py --compare_to_retriever

# Fine-tune (optional, ~90 min on CPU / ~10 min on GPU)
python reranker/train_reranker.py --epochs 1
python scripts/evaluate_reranker.py \
    --model reranker/checkpoints/finetuned \
    --run_name ms_marco_finetuned \
    --compare_to_retriever
```

**Router** (Person C's component — query classifier):

```bash
cd router
python label.py                # Generate weak-supervision labels
python train.py                # Train logistic regression
python evaluate.py             # Validation accuracy + plots
cd ..
```

**View consolidated results:**

```bash
python -c "from eval_harness.runner import print_leaderboard; print_leaderboard()"
```

---

## Method Details

### Dataset

**MedHop** (Welbl et al., 2018) from the QAngaroo benchmark:
- 1,620 training examples · 342 validation examples
- ~30 candidate support passages per question
- Queries of the form `interacts_with DBXXXXX?` with DrugBank IDs
- Multi-hop by design — answers require chaining evidence across passages

Loaded via the HuggingFace `datasets` library (`qangaroo/medhop`).

### Retriever — Fine-tuned MedCPT

Asymmetric bi-encoder built on `ncbi/MedCPT-Query-Encoder` and `ncbi/MedCPT-Article-Encoder`. Fine-tuned with InfoNCE loss using in-batch negatives plus pre-mined BM25 hard negatives. Closed-domain Recall@5 improves from 0.246 (zero-shot) to 0.412 (fine-tuned), a +16.6 pp lift.

### Reranker — Cross-Encoder

`cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top candidates from the retriever. Unlike the bi-encoder, it concatenates query and passage into a single transformer input, allowing full cross-attention at the cost of inference speed. Used both zero-shot and fine-tuned on MedHop with a pairwise margin objective.

### Router — Weak-Supervision Classifier

Binary logistic regression over 5 structural features (passage count, candidate count, BM25 top score, score gap, candidate-passage overlap). Labels derived via BM25 answer-surfacing median split, since MedHop provides no ground-truth hop-count annotations. The classifier is small enough to evaluate at near-zero inference cost.

---

## Design Decisions

- **Closed-domain evaluation.** MedHop ships each question with its own support set. We rank within that set rather than over the global corpus, because the global corpus uses drug names while answers are DrugBank IDs — making global string-match relevance signal nearly nonexistent. Closed-domain evaluation matches how the retriever is actually used in the pipeline (rank a pre-fetched candidate pool).

- **Binary router instead of 3-way.** The original proposal specified a single/2-hop/3-hop classifier, but BM25-derived weak labels cannot reliably distinguish 2-hop from 3-hop. Reduced to binary single-vs-multi.

- **Cross-encoder reranker over bi-encoder reranker.** A second bi-encoder would not add complementary signal. Cross-encoders are slower per pair but only see 30 candidates per query — full cross-attention is computationally feasible.

- **Shared `results.json` over W&B.** A single merged JSON file with a `append_to_leaderboard()` helper avoided third-party setup and made results trivially diffable.

---

## Limitations and Future Work

- **MedHop only.** The original proposal included MedQA and MedHopQA. These were cut to scope; the architecture supports them with weak-supervision labels (entity overlap for MedQA, concept-level scoring for MedHopQA).
- **End-to-end QA evaluation.** The agent loop and final EM/concept-score evaluation require integration of all four components; this is in-progress.
- **Router features.** The full proposed feature set required dense retriever scores (BM25-dense gap, top-k entropy, sparse-dense agreement). The current router uses 5 structural features only; v2 features depend on retriever availability at training time.

---

## Team

| Component | Owner |
|-----------|-------|
| Retriever | Carol |
| Reranker + Eval Harness | Behjat|
| Router | Selina |
| Agent + Integration | Ken |

---

## References

- Welbl, J., Stenetorp, P., & Riedel, S. (2018). *Constructing Datasets for Multi-hop Reading Comprehension Across Documents.* TACL.
- Jin, Q. et al. (2023). *MedCPT: Contrastive Pre-trained Transformers for Information Retrieval.* Bioinformatics.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Trivedi, H. et al. (2023). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions.* ACL.

---

## License

Academic use only. CS 572, Spring 2026, Emory University.
