# Reranker + Evaluation Harness — Handoff

**Owner:** Person B
**Branch:** `person-b/reranker-and-eval`
**Status:** Shipped — zero-shot and fine-tuned cross-encoder + shared eval harness
**Version:** 1.0

For Person C (router) and Person D (agent/integration). The interfaces below are working — you can import and call them today.

---

## Headline Result

The fine-tuned cross-encoder reranker improves over the best retriever-only baseline (Person A's fine-tuned MedCPT) by **+15.5 pp on Recall@5** and **+17.5 pp on Recall@10** on the MedHop validation split.

| Model | R@1 | R@5 | R@10 | nDCG@5 | nDCG@10 |
|-------|----:|----:|-----:|-------:|--------:|
| BM25 | 0.041 | 0.199 | 0.409 | 0.110 | 0.176 |
| Vanilla MedCPT | 0.059 | 0.246 | 0.491 | 0.137 | 0.215 |
| Fine-tuned MedCPT (Person A) | 0.082 | 0.412 | 0.693 | 0.230 | 0.316 |
| Zero-shot reranker | 0.041 | 0.336 | 0.585 | 0.172 | 0.248 |
| **Fine-tuned reranker** | **0.158** | **0.567** | **0.868** | **0.343** | **0.446** |

---

## What's in this branch

```
reranker/
  reranker.py              # public interface — import Reranker / rerank() from here
  train_reranker.py        # fine-tuning script (pairwise margin loss)
  requirements.txt         # pip deps
eval_harness/
  metrics.py               # recall_at_k, ndcg_at_k, map_score, reciprocal_rank, exact_match
  runner.py                # evaluate_rankings(), append_to_leaderboard(), print_leaderboard()
scripts/
  evaluate_reranker.py     # reproduce the reranker eval on MedHop val
results.json               # shared — reranker numbers under "reranker" key
```

---

## For Person D (agent loop + integration)

### Adding the reranker to your pipeline

```python
from reranker import rerank

# After Person A's retriever returns candidates, rerank them
retrieved = retriever.retrieve(query, k=30)   # ~30 candidate passages
top_passages = rerank(query, retrieved, top_k=5)
# top_passages is [(passage_text, score), ...] sorted best-first

context = "\n\n".join(p for p, _ in top_passages)
answer = generator(query, context)
```

**Recommended:** `N_retrieve = 30`, `top_k = 5`. Lower N starves the reranker; higher N wastes compute with diminishing returns.

**Use the fine-tuned model in production.** The default `rerank()` function loads the zero-shot ms-marco model. To use the fine-tuned weights:

```python
from reranker import Reranker
rr = Reranker(model_name="reranker/checkpoints/finetuned")
top_passages = rr.rerank(query, retrieved, top_k=5)
```

The fine-tuned checkpoint is gitignored — regenerate with `python reranker/train_reranker.py --epochs 1`.

**For multi-hop retrieval:** call `rerank()` after each retrieval hop. The scores it returns can feed your early-stopping logic — large score gaps between top-1 and top-2 signal high confidence.

**Persistent model:** the first call loads the model (~1-2 s); subsequent calls reuse it.

### Using the eval harness for end-to-end QA evaluation

```python
from eval_harness.metrics import exact_match
from eval_harness.runner import append_to_leaderboard

em_scores = [exact_match(pred, gold) for pred, gold in zip(predictions, golds)]
em = sum(em_scores) / len(em_scores)

append_to_leaderboard("end_to_end", "agent_with_router_v1",
                     {"exact_match": em, "n_examples": len(predictions)})
```

For retrieval-style metrics (if you're evaluating the full pipeline's retrieval quality):

```python
from eval_harness.runner import evaluate_rankings

# rankings: {question_id: [passage_id, ...]}
# relevants: {question_id: {passage_id, ...}}
metrics = evaluate_rankings(rankings, relevants, k_values=(1, 5, 10))
```

---

## For Person C (router)

Your router is already trained and working. Two optional hooks if you have cycles:

**1. Move to the shared metrics.** Right now your `router/evaluate.py` reimplements accuracy/confusion matrix. If you switch to `eval_harness` helpers, your numbers live in the shared `results.json`:

```python
from eval_harness.runner import append_to_leaderboard
append_to_leaderboard("router", "lr_5feat_v1",
                     {"accuracy": 0.693, "cv_mean": ..., "n_val": 342})
```

**2. v2 features that need the dense retriever.** Your scope-reduction note says BM25-dense gap, score entropy, and sparse-dense agreement were dropped because the retriever didn't exist yet. Now it does. If you want to add those features, ping me and I'll write a helper that exposes retriever scores on your support passages.

---

## Shared `results.json` schema

Everyone writes to the same file via `append_to_leaderboard()`. **Never overwrite** — the helper merges.

```json
{
  "retriever": {
    "bm25":             { "recall@5": 0.199, ... },
    "medcpt_finetuned": { "recall@5": 0.412, "ndcg@5": 0.230, ... }
  },
  "reranker": {
    "ms_marco_zero_shot": { "recall@5": 0.336, "ndcg@5": 0.172, ... },
    "ms_marco_finetuned": { "recall@5": 0.567, "ndcg@5": 0.343, ... }
  },
  "router": {
    "lr_5feat_v1": { "accuracy": 0.693, ... }
  },
  "end_to_end": {
    "full_pipeline_v1": { ... }
  }
}
```

Print everything with:

```bash
python -c "from eval_harness.runner import print_leaderboard; print_leaderboard()"
```

---

## Reproducing my numbers

```bash
# Install dependencies
pip install -r reranker/requirements.txt

# Zero-shot eval (~15 min CPU, ~2 min GPU)
python scripts/evaluate_reranker.py --compare_to_retriever

# Fine-tune (~90 min CPU, ~10 min GPU)
python reranker/train_reranker.py --epochs 1

# Fine-tuned eval
python scripts/evaluate_reranker.py \
    --model reranker/checkpoints/finetuned \
    --run_name ms_marco_finetuned \
    --compare_to_retriever
```

Assumes Person A's `retriever/data/corpus.json` and `retriever/data/val_triples.json` exist (run `retriever/data_prep.py` if not).

---

## Method Summary

**Architecture.** Cross-encoder built on `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params). Concatenates query and passage as `[CLS] query [SEP] passage [SEP]` and outputs a single relevance score per pair. Slower per pair than a bi-encoder, but only sees the top 30 candidates per query.

**Training.** Pairwise objective using BCE on (positive=1, hard_negative=0) pairs. Training data reuses Person A's `train_triples_with_negatives.json` — positives from the `positives` field, hard negatives from the `hard_negatives` field. One epoch over 1,620 MedHop questions.

**Evaluation protocol.** Closed-domain per-example, identical to Person A's retriever evaluation. Each query's ~30 support passages are reranked among themselves; relevant set = passages containing the answer drug-ID as a substring; metrics are Recall@K, nDCG@K, MAP, MRR.

---

## Known Limitations / Future Work

- **MedQA weak supervision: not shipped.** Originally scoped but cut from MVP (dataset not used in final project). Pseudo-label generation via entity overlap is documented in the project proposal.
- **MedHopQA concept-level score: not shipped.** Same reason — dataset not used.
- **Eval harness is retrieval-metric-heavy.** EM is the only QA metric currently. Concept score would need a UMLS mapping layer; add if MedHopQA is reintroduced.
- **One epoch only.** More epochs may yield further gains but were not run due to time. The current result already exceeds the retriever-only baseline by 15.5 pp.

---

## Questions?

Ping me on the branch or open an issue tagged `Behjat4138`.