"""Deterministic stratified evaluation set sampler for QASPER.

Samples a fixed evaluation set (default 200 questions) from the validation
split, stratified across question answer types (unanswerable, yes/no,
extractive, free-form) to preserve exact dataset proportions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from researchmate.data.models import Answer, QAExample

log = logging.getLogger(__name__)


def categorize_qa(qa: QAExample | dict[str, Any]) -> str:
    """Categorize a QA item into mutually exclusive answer types.

    Precedence order aligns with QASPER evaluation protocols:
        1. ``unanswerable``: at least one annotator marked unanswerable
        2. ``yes_no``: at least one annotator provided a boolean yes/no
        3. ``extractive``: answers contain extractive text spans
        4. ``free_form``: answers contain free-form paraphrase text
        5. ``other``: fallback for unclassified questions

    Args:
        qa: A QAExample model instance or raw QA dict.

    Returns:
        One of ``'unanswerable'``, ``'yes_no'``, ``'extractive'``,
        ``'free_form'``, ``'other'``.
    """
    if isinstance(qa, QAExample):
        answers: list[Answer] = qa.answers
        if any(a.unanswerable for a in answers):
            return "unanswerable"
        if any(a.yes_no is not None for a in answers):
            return "yes_no"
        if any(bool(a.extractive_spans) for a in answers):
            return "extractive"
        if any(bool(a.free_form) for a in answers):
            return "free_form"
        return "other"

    raw_answers = qa.get("answers", [])
    for annotator in raw_answers:
        raw_ans = annotator.get("answer", annotator) if isinstance(annotator, dict) else {}
        if raw_ans.get("unanswerable"):
            return "unanswerable"
    for annotator in raw_answers:
        raw_ans = annotator.get("answer", annotator) if isinstance(annotator, dict) else {}
        if raw_ans.get("yes_no") is not None:
            return "yes_no"
    for annotator in raw_answers:
        raw_ans = annotator.get("answer", annotator) if isinstance(annotator, dict) else {}
        if raw_ans.get("extractive_spans"):
            return "extractive"
    for annotator in raw_answers:
        raw_ans = annotator.get("answer", annotator) if isinstance(annotator, dict) else {}
        if raw_ans.get("free_form_answer"):
            return "free_form"
    return "other"


def sample_stratified_qa(
    qa_list: list[dict[str, Any]],
    n: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample ``n`` questions using deterministic stratified sampling.

    Uses the largest remainder method (Hare-Niemeyer / Hamilton quota) to
    allocate integer quotas per stratum that sum to exactly ``n``.
    Items are sorted deterministically before sampling with a seeded RNG.

    Args:
        qa_list: List of QA candidate dictionaries.
        n: Total number of questions to sample.
        seed: Random seed for deterministic reproducibility.

    Returns:
        A tuple of ``(sampled_questions, quotas_per_stratum)``.
    """
    if not qa_list:
        raise ValueError("Cannot sample from an empty QA list.")
    if n > len(qa_list):
        raise ValueError(f"Requested sample size {n} exceeds population size {len(qa_list)}.")

    # Group by category
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for qa in qa_list:
        cat = qa.get("category") or categorize_qa(qa)
        by_cat.setdefault(cat, []).append(qa)

    # Sort each stratum deterministically by question_id and paper_id
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: (str(x.get("question_id", "")), str(x.get("paper_id", ""))))

    # Compute quotas using largest remainder method
    total_pop = len(qa_list)
    raw_quotas = {cat: (len(items) * n) / total_pop for cat, items in by_cat.items()}
    int_quotas = {cat: int(math.floor(q)) for cat, q in raw_quotas.items()}
    remainders = {cat: raw_quotas[cat] - int_quotas[cat] for cat in by_cat}

    missing = n - sum(int_quotas.values())
    sorted_remainders = sorted(remainders.items(), key=lambda x: (-x[1], x[0]))
    for cat, _ in sorted_remainders[:missing]:
        int_quotas[cat] += 1

    # Sample from each stratum deterministically
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for cat in sorted(int_quotas.keys()):
        quota = int_quotas[cat]
        chosen = rng.sample(by_cat[cat], quota)
        sampled.extend(chosen)

    # Sort final sample deterministically
    sampled.sort(key=lambda x: (str(x.get("question_id", "")), str(x.get("paper_id", ""))))

    assert len(sampled) == n, f"Expected {n} sampled questions, got {len(sampled)}"
    return sampled, int_quotas


def create_and_save_eval_set(
    val_jsonl_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    n: int = 200,
    seed: int = 42,
) -> tuple[Path, str, dict[str, int]]:
    """Load validation papers, sample stratified questions, and save eval_set.jsonl.

    Also calculates SHA-256 and updates the manifest if a manifest_path is provided.

    Args:
        val_jsonl_path: Path to ``data/raw/validation.jsonl``.
        output_path: Destination path for ``data/eval/eval_set.jsonl``.
        manifest_path: Optional path to ``MANIFEST.json`` to update.
        n: Number of questions to sample.
        seed: Random seed.

    Returns:
        A tuple of ``(output_path, sha256_hex, stratum_quotas)``.
    """
    if not val_jsonl_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_jsonl_path}")

    qa_candidates: list[dict[str, Any]] = []
    with val_jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            paper_id = paper.get("paper_id", "")
            title = paper.get("title", "")
            for qa in paper.get("qas", []):
                cat = categorize_qa(qa)
                qa_candidates.append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "question_id": qa.get("question_id", ""),
                        "question": qa.get("question", ""),
                        "category": cat,
                        "answers": [
                            a.get("answer", a) for a in qa.get("answers", []) if isinstance(a, dict)
                        ],
                    }
                )

    sampled, quotas = sample_stratified_qa(qa_candidates, n=n, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for item in sampled:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Compute SHA-256
    content = output_path.read_bytes()
    sha256_hex = hashlib.sha256(content).hexdigest()

    log.info(
        "Saved %d stratified questions to %s (SHA-256: %s)",
        len(sampled),
        output_path,
        sha256_hex,
    )

    # Update manifest if provided
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "splits" not in manifest_data:
                manifest_data["splits"] = {}
            manifest_data["splits"]["eval"] = {
                "file": output_path.name,
                "rows": len(sampled),
                "size_bytes": len(content),
                "sha256": sha256_hex,
                "sample_seed": seed,
                "stratification_quotas": quotas,
            }
            manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
            log.info("Updated %s with eval split checksum.", manifest_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not update manifest at %s: %s", manifest_path, exc)

    return output_path, sha256_hex, quotas
