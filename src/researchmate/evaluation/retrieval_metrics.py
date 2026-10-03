"""Information retrieval evaluation metrics: Hit Rate, Recall, MRR, and NDCG."""

from __future__ import annotations

import math


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: set[str] | list[str], k: int = 5) -> float:
    """Compute HitRate@k: 1.0 if at least one gold ID appears in top-k, else 0.0."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(cid in gold_set for cid in top_k) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str] | list[str], k: int = 5) -> float:
    """Compute Recall@k: fraction of gold IDs retrieved in top-k."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_k = set(retrieved_ids[:k])
    return len(gold_set & top_k) / len(gold_set)


def mrr_at_k(retrieved_ids: list[str], gold_ids: set[str] | list[str], k: int = 5) -> float:
    """Compute Mean Reciprocal Rank (MRR@k): 1 / rank of first gold ID in top-k."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str] | list[str], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@k) with binary relevance."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in gold_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all gold items ranked at the top, up to k
    n_ideal = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, n_ideal + 1))

    return dcg / idcg if idcg > 0 else 0.0
