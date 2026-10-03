"""Unit tests for deterministic evaluation set sampling and categorization.

Tests cover:
- Categorization logic for Answer and QAExample models and raw dicts
- Exact quota distribution using largest remainder method
- Deterministic reproducibility: same seed gives identical samples
- Seed sensitivity: different seeds give different samples
- Input validation and boundary error handling
- Mini fixture sampling and file generation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchmate.data.models import Answer, QAExample
from researchmate.data.sampler import (
    categorize_qa,
    create_and_save_eval_set,
    sample_stratified_qa,
)

# =========================================================================
# 1. Categorization Tests
# =========================================================================


def test_categorize_unanswerable_priority() -> None:
    """Unanswerable takes priority over other answer types."""
    qa = QAExample(
        paper_id="p1",
        question_id="q1",
        question="What is this?",
        answers=[
            Answer(unanswerable=True),
            Answer(extractive_spans=["some text"]),
        ],
    )
    assert categorize_qa(qa) == "unanswerable"


def test_categorize_yes_no() -> None:
    """Yes/no questions are identified correctly."""
    qa = QAExample(
        paper_id="p1",
        question_id="q2",
        question="Is it true?",
        answers=[Answer(yes_no=True)],
    )
    assert categorize_qa(qa) == "yes_no"


def test_categorize_extractive() -> None:
    """Extractive span answers are identified correctly."""
    qa = QAExample(
        paper_id="p1",
        question_id="q3",
        question="Which model?",
        answers=[Answer(extractive_spans=["BERT-base"])],
    )
    assert categorize_qa(qa) == "extractive"


def test_categorize_free_form() -> None:
    """Free form paraphrase answers are identified correctly."""
    qa = QAExample(
        paper_id="p1",
        question_id="q4",
        question="How does it work?",
        answers=[Answer(free_form="It works by propagating labels.")],
    )
    assert categorize_qa(qa) == "free_form"


def test_categorize_from_raw_dict() -> None:
    """categorize_qa works on raw dictionary format from JSONL."""
    raw_unans = {"answers": [{"answer": {"unanswerable": True}}]}
    raw_yesno = {"answers": [{"answer": {"yes_no": False}}]}
    raw_ext = {"answers": [{"answer": {"extractive_spans": ["abc"]}}]}
    raw_ff = {"answers": [{"answer": {"free_form_answer": "paraphrase"}}]}
    raw_empty = {"answers": []}

    assert categorize_qa(raw_unans) == "unanswerable"
    assert categorize_qa(raw_yesno) == "yes_no"
    assert categorize_qa(raw_ext) == "extractive"
    assert categorize_qa(raw_ff) == "free_form"
    assert categorize_qa(raw_empty) == "other"


# =========================================================================
# 2. Stratified Sampling Determinism & Quotas
# =========================================================================


def _make_population() -> list[dict]:
    """Create a synthetic population of 100 questions across 4 categories."""
    pop = []
    # 60 extractive, 20 free_form, 10 unanswerable, 10 yes_no
    for i in range(60):
        pop.append(
            {"question_id": f"q_ext_{i:02d}", "paper_id": f"p_{i}", "category": "extractive"}
        )
    for i in range(20):
        pop.append({"question_id": f"q_ff_{i:02d}", "paper_id": f"p_{i}", "category": "free_form"})
    for i in range(10):
        pop.append(
            {"question_id": f"q_un_{i:02d}", "paper_id": f"p_{i}", "category": "unanswerable"}
        )
    for i in range(10):
        pop.append({"question_id": f"q_yn_{i:02d}", "paper_id": f"p_{i}", "category": "yes_no"})
    return pop


def test_sampler_is_deterministic() -> None:
    """Running sampler twice with the same seed yields identical results."""
    pop = _make_population()
    sample1, quotas1 = sample_stratified_qa(pop, n=20, seed=42)
    sample2, quotas2 = sample_stratified_qa(pop, n=20, seed=42)

    assert quotas1 == quotas2
    assert len(sample1) == 20
    assert len(sample2) == 20
    assert [x["question_id"] for x in sample1] == [x["question_id"] for x in sample2]


def test_sampler_different_seeds_diverge() -> None:
    """Different random seeds yield different samples."""
    pop = _make_population()
    sample1, _ = sample_stratified_qa(pop, n=20, seed=42)
    sample2, _ = sample_stratified_qa(pop, n=20, seed=999)

    assert [x["question_id"] for x in sample1] != [x["question_id"] for x in sample2]


def test_sampler_quotas_match_proportions() -> None:
    """Sample quotas reflect population proportions."""
    pop = _make_population()  # 60% ext, 20% ff, 10% un, 10% yn
    sample, quotas = sample_stratified_qa(pop, n=20, seed=42)

    assert len(sample) == 20
    # Expected: 60% of 20 = 12, 20% of 20 = 4, 10% of 20 = 2, 10% of 20 = 2
    assert quotas["extractive"] == 12
    assert quotas["free_form"] == 4
    assert quotas["unanswerable"] == 2
    assert quotas["yes_no"] == 2
    assert sum(quotas.values()) == 20


def test_sampler_boundary_errors() -> None:
    """Sampler raises appropriate ValueErrors on invalid arguments."""
    with pytest.raises(ValueError, match="Cannot sample from an empty QA list"):
        sample_stratified_qa([], n=10)

    pop = _make_population()
    with pytest.raises(ValueError, match="exceeds population size"):
        sample_stratified_qa(pop, n=200)


# =========================================================================
# 3. File Creation & Manifest Update
# =========================================================================


def test_create_and_save_eval_set_on_fixture(tmp_path: Path) -> None:
    """Test creating an evaluation set file and updating manifest on fixture."""
    fixture_path = Path("tests/fixtures/mini_qasper.json")
    out_file = tmp_path / "eval_set.jsonl"
    manifest_file = tmp_path / "MANIFEST.json"
    manifest_file.write_text(json.dumps({"source": "test", "splits": {}}), encoding="utf-8")

    # Fixture has 8 questions total across 3 papers
    saved_path, sha256_hex, quotas = create_and_save_eval_set(
        val_jsonl_path=fixture_path,
        output_path=out_file,
        manifest_path=manifest_file,
        n=4,
        seed=42,
    )

    assert saved_path.exists()
    assert len(sha256_hex) == 64
    assert sum(quotas.values()) == 4

    lines = [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 4
    assert all("question_id" in row and "paper_id" in row and "category" in row for row in lines)

    # Verify manifest was updated
    manifest_data = json.loads(manifest_file.read_text())
    assert "eval" in manifest_data["splits"]
    assert manifest_data["splits"]["eval"]["rows"] == 4
    assert manifest_data["splits"]["eval"]["sha256"] == sha256_hex
