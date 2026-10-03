"""Unit tests for gold evidence to chunk mapping.

Tests cover:
- Text normalization for evidence matching
- Exact substring matching against raw/cleaned paragraphs
- Fuzzy sequence matching above threshold
- Mapping paragraph matches to parent Chunk IDs
- Handling unmatched float (table/figure) evidence with explicit reasons
- Fixture paper evidence mapping
"""

from __future__ import annotations

from pathlib import Path

from researchmate.chunking.chunker import chunk_paper
from researchmate.chunking.evidence import (
    map_evidence_to_chunks,
    match_evidence_to_paragraphs,
    normalize_for_matching,
)
from researchmate.data.loader import load_papers
from researchmate.data.models import Paper, Paragraph

# =========================================================================
# 1. Normalization Tests
# =========================================================================


def test_normalize_for_matching() -> None:
    """normalize_for_matching handles case, punctuation, and whitespace."""
    raw = "  FLOAT SELECTED: Table 1: Performance (accuracy = 85.5%)!  "
    norm = normalize_for_matching(raw)
    assert norm == "table 1 performance accuracy 85 5"
    assert normalize_for_matching("") == ""


# =========================================================================
# 2. Paragraph Matching Tests
# =========================================================================


def test_match_evidence_exact() -> None:
    """Exact substring matches return the correct paragraph index with score 1.0."""
    p0 = Paragraph(paper_id="p1", section="Intro", index=0, text="First introductory note.")
    p1 = Paragraph(
        paper_id="p1",
        section="Intro",
        index=1,
        text="Affective events are events that typically affect people in positive ways.",
    )
    paragraphs = [p0, p1]

    evidence = "Affective events are events that typically affect people"
    matches, score, match_type = match_evidence_to_paragraphs(evidence, paragraphs)

    assert matches == [1]
    assert score == 1.0
    assert match_type == "exact"


def test_match_evidence_fuzzy() -> None:
    """Fuzzy matching identifies the paragraph despite slight word modifications."""
    p0 = Paragraph(paper_id="p1", section="Intro", index=0, text="Unrelated text about biology.")
    p1 = Paragraph(
        paper_id="p1",
        section="Methods",
        index=1,
        text="We trained a deep bidirectional transformer model on the English corpus.",
    )
    paragraphs = [p0, p1]

    # Slightly modified evidence with minor typos or missing words
    evidence = "trained deep bidirectional transformer model on English corpus"
    matches, score, match_type = match_evidence_to_paragraphs(
        evidence, paragraphs, fuzzy_threshold=0.75
    )

    assert matches == [1]
    assert score >= 0.75
    assert match_type in ("exact", "fuzzy")


def test_match_evidence_none() -> None:
    """Unmatched evidence returns empty matches and match_type 'none'."""
    p0 = Paragraph(paper_id="p1", section="Intro", index=0, text="Some text about robotics.")
    paragraphs = [p0]

    evidence = "Astronomical observation of distant galaxies."
    matches, score, match_type = match_evidence_to_paragraphs(evidence, paragraphs)

    assert matches == []
    assert match_type == "none"
    assert score < 0.80


# =========================================================================
# 3. Chunk Mapping and Float Evidence Handling
# =========================================================================


def test_map_evidence_to_chunks() -> None:
    """map_evidence_to_chunks correctly associates matched paragraphs with chunks."""
    paper = Paper(
        paper_id="p1",
        title="Test Paper",
        paragraphs=[
            Paragraph(paper_id="p1", section="Intro", index=0, text="Para 0 intro."),
            Paragraph(
                paper_id="p1",
                section="Intro",
                index=1,
                text="BERT is an effective language representation model.",
            ),
            Paragraph(paper_id="p1", section="Methods", index=2, text="Para 2 methods."),
        ],
    )

    chunks = chunk_paper(paper, max_tokens=100)
    evidence_strings = [
        "BERT is an effective language representation model.",
        "FLOAT SELECTED: Table 2: Model comparison results on GLUE benchmark.",
    ]

    mapping = map_evidence_to_chunks(evidence_strings, chunks, paper.paragraphs)

    assert len(mapping["gold_chunk_ids"]) > 0
    # First evidence is matched to a chunk
    assert mapping["details"][0]["matched"] is True
    assert len(mapping["details"][0]["matched_chunks"]) > 0

    # Second evidence is float table -> unmatched with reason
    assert mapping["details"][1]["matched"] is False
    assert mapping["details"][1]["unmatched_reason"] == "figure_or_table_evidence"


def test_evidence_mapping_on_fixture() -> None:
    """Verify evidence mapping on fixture papers."""
    papers, qa_list = load_papers(Path("tests/fixtures/mini_qasper.json"))
    paper = papers[0]
    chunks = chunk_paper(paper)

    # Collect any evidence from questions about this paper
    paper_qas = [q for q in qa_list if q.paper_id == paper.paper_id]
    ev_strings = []
    for q in paper_qas:
        for a in q.answers:
            ev_strings.extend(a.evidence)

    if ev_strings:
        mapping = map_evidence_to_chunks(ev_strings, chunks, paper.paragraphs)
        assert "gold_chunk_ids" in mapping
        assert "details" in mapping
