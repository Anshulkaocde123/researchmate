# Retrieval Performance Evaluation Report

- **Evaluation Dataset:** `data/eval/eval_with_gold_chunks.jsonl`
- **Questions Evaluated:** 25
- **Evaluation Scope:** Paper-scoped retrieval (strict paper_id filter)

## Overall Metrics Comparison

| Metric | BM25 (Sparse) | Hybrid (RRF + Rerank) |
|---|:---:|:---:|
| **HitRate@1** | 32.0% | 28.0% |
| **HitRate@3** | 60.0% | 60.0% |
| **HitRate@5** | 88.0% | 84.0% |
| **HitRate@10** | 92.0% | 92.0% |
| **HitRate@20** | 96.0% | 96.0% |
| **MRR@5** | 0.500 | 0.483 |
| **MRR@10** | 0.505 | 0.494 |
| **NDCG@5** | 0.492 | 0.513 |
| **NDCG@10** | 0.571 | 0.572 |

## Stratified Performance by Question Type (Hybrid @ k=5)

| Question Type | Count | HitRate@5 | Recall@5 | MRR@5 |
|---|:---:|:---:|:---:|:---:|
| Extractive | 22 | 81.8% | 69.0% | 0.483 |
| Free_form | 1 | 100.0% | 40.0% | 1.000 |
| Unanswerable | 1 | 100.0% | 100.0% | 0.200 |
| Yes_no | 1 | 100.0% | 50.0% | 0.250 |
