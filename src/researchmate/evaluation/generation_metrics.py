"""Generation and citation evaluation metrics: Token F1, Exact Match, and Citation Precision/Recall."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute binary Exact Match score between prediction and ground truth."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def citation_metrics(
    cited_chunk_ids: list[str],
    gold_chunk_ids: set[str] | list[str],
) -> dict[str, float]:
    """Compute citation precision, recall, and F1 relative to gold evidence chunks.

    Args:
        cited_chunk_ids: Chunk IDs explicitly cited in the generated answer.
        gold_chunk_ids: Ground truth chunk IDs for the question.

    Returns:
        Dict with precision, recall, f1.
    """
    cited_set = set(cited_chunk_ids)
    gold_set = set(gold_chunk_ids)

    if not cited_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not cited_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    intersection = cited_set & gold_set
    precision = len(intersection) / len(cited_set)
    recall = len(intersection) / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
