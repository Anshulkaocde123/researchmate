"""Unit tests for text cleaning and artifact removal.

Covers:
- Parametrized tests for each documented artifact pattern
- Mathematical idempotency property: clean_text(clean_text(x)) == clean_text(x)
- Paragraph and Paper level cleaning preserving raw_text
- Evidence string cleaning (FLOAT SELECTED prefixes)
- Real fixture paragraphs from tests/fixtures/mini_qasper.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchmate.data.cleaning import (
    clean_evidence,
    clean_paper,
    clean_paragraph,
    clean_text,
)
from researchmate.data.loader import load_papers
from researchmate.data.models import Paper, Paragraph

# =========================================================================
# 1. Parametrized Tests for Every Documented Pattern
# =========================================================================


@pytest.mark.parametrize(
    ("raw", "expected_clean"),
    [
        # BIBREF variations
        ("Affective events BIBREF0 are events.", "Affective events are events."),
        ("Models (BIBREF1, BIBREF2) were tested.", "Models were tested."),
        ("See [BIBREF0] and [BIBREF1; BIBREF2].", "See and."),
        ("Approaches (e.g., BIBREF0; see BIBREF1).", "Approaches."),
        ("Prefix token BERTBIBREF0 is stripped.", "Prefix token is stripped."),
        # FLOAT SELECTED variations
        (
            "FLOAT SELECTED: Table 3: Performance of various models.",
            "Table 3: Performance of various models.",
        ),
        (
            "float selected: Figure 1: Model architecture diagram.",
            "Figure 1: Model architecture diagram.",
        ),
        ("FLOAT SELECTED Table 4: Results.", "Table 4: Results."),
        # Structural reference tokens
        (
            "As shown in Table TABREF16 and Figure FIGREF1.",
            "As shown in Table and Figure.",
        ),
        ("In Section SECREF21, we describe...", "In Section, we describe..."),
        ("Defined by Eq ( EQREF13 ).", "Defined by Eq."),
        ("Found in UNKREF6.", "Found in."),
        # Formula placeholders
        (
            "According to INLINEFORM0 and DISPLAYFORM1.",
            "According to and.",
        ),
        # LaTeX math inline and display
        ("Score ranging from $-1$ to $+1$.", "Score ranging from -1 to +1."),
        ("Let variable $x_1$ equal $y_2$.", "Let variable x_1 equal y_2."),
        ("Display $$E = mc^2$$ formula.", "Display E = mc^2 formula."),
        # LaTeX font commands & formatting
        ("${\\rm Encoder}$ outputs a representation.", "Encoder outputs a representation."),
        ("We use \\textbf{BERT} and \\textit{RoBERTa}.", "We use BERT and RoBERTa."),
        ("Features \\mathrm{feat} and \\mathbf{x}.", "Features feat and x."),
        ("A {\\bf bold} statement.", "A bold statement."),
        # LaTeX symbols & escapes
        ("Values $i=1, \\cdots, n$.", "Values i=1, ..., n."),
        ("Dimensions $3 \\times 4$.", "Dimensions 3 x 4."),
        ("Confidence $p \\pm 0.05$.", "Confidence p +/- 0.05."),
        ("Bounds $a \\leq b$ and $b \\geq c$.", "Bounds a <= b and b >= c."),
        ("Check $x \\neq y$ and $x \\approx z$.", "Check x != y and x ~ z."),
        ("Increased by 15\\% and costs \\$10.", "Increased by 15% and costs $10."),
        (
            "Cost is \\$100 with 20\\% tax on items \\#1 \\& \\#2.",
            "Cost is $100 with 20% tax on items #1 & #2.",
        ),
        ("Code in under\\_score and key\\&value.", "Code in under_score and key&value."),
        # Odd whitespace & newlines
        ("Line 1\nLine 2\r\nLine 3\tTabbed.", "Line 1 Line 2 Line 3 Tabbed."),
        ("Multiple   consecutive    spaces.", "Multiple consecutive spaces."),
        ("  Leading and trailing whitespace.  ", "Leading and trailing whitespace."),
        # Orphaned brackets & duplicate punctuation
        ("Empty parens () and brackets [].", "Empty parens and brackets."),
        ("Residual comma , and space before dot .", "Residual comma, and space before dot."),
        ("Duplicate commas ,, and dots .. here.", "Duplicate commas, and dots. here."),
        (
            "Ellipsis like .... should be three dots ...",
            "Ellipsis like ... should be three dots ...",
        ),
    ],
)
def test_clean_text_patterns(raw: str, expected_clean: str) -> None:
    """Verify clean_text removes every targeted artifact correctly."""
    assert clean_text(raw) == expected_clean


# =========================================================================
# 2. Idempotency Tests: clean_text(clean_text(x)) == clean_text(x)
# =========================================================================


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "Already clean plain text without any artifacts.",
        "Affective events BIBREF0 are events that typically affect people.",
        "FLOAT SELECTED: Table 3: Performance of various models on the ACP test set.",
        "${\\rm Encoder}$ outputs a vector representation of the event $x$.",
        "In Section SECREF21, we consider two specific implementations.",
        "Figure FIGREF1 and Table TABREF16 (see BIBREF0, BIBREF1; ) show results.",
        "Line 1\n\nLine 2\r\nLine 3 with   extra    spaces.",
        "Formula INLINEFORM4 according to Eq ( EQREF10 ) and DISPLAYFORM0.",
        "Double punctuation (,,  ..) and trailing dots....",
        "Cost is \\$100 with 20\\% tax on items \\#1 \\& \\#2.",
    ],
)
def test_clean_text_idempotence(raw: str) -> None:
    """Ensure clean_text satisfies idempotency: clean(clean(x)) == clean(x)."""
    cleaned_once = clean_text(raw)
    cleaned_twice = clean_text(cleaned_once)
    assert cleaned_twice == cleaned_once


def test_clean_text_idempotence_on_fixture() -> None:
    """Ensure idempotency holds across all paragraphs in the mini_qasper fixture."""
    fixture_path = Path("tests/fixtures/mini_qasper.json")
    papers = [
        Paper.model_validate(json.loads(line)) for line in fixture_path.read_text().splitlines()
    ]

    for paper in papers:
        for para in paper.paragraphs:
            c1 = clean_text(para.text)
            c2 = clean_text(c1)
            assert c1 == c2, f"Idempotency failed on paper {paper.paper_id}, para {para.index}"


# =========================================================================
# 3. Model Integration & Raw Text Preservation
# =========================================================================


def test_paragraph_auto_populates_raw_text() -> None:
    """Paragraph automatically initializes raw_text to text if not explicitly set."""
    p = Paragraph(
        paper_id="2002.06424",
        section="Intro",
        index=0,
        text="Raw text with BIBREF0.",
    )
    assert p.raw_text == "Raw text with BIBREF0."
    assert p.text == "Raw text with BIBREF0."


def test_clean_paragraph_preserves_raw_text() -> None:
    """clean_paragraph cleans text while keeping original raw_text intact."""
    raw = "Affective events BIBREF0 are positive."
    p = Paragraph(
        paper_id="2002.06424",
        section="Intro",
        index=0,
        text=raw,
    )
    cleaned_p = clean_paragraph(p)
    assert cleaned_p.text == "Affective events are positive."
    assert cleaned_p.raw_text == raw
    assert cleaned_p.paper_id == p.paper_id
    assert cleaned_p.section == p.section
    assert cleaned_p.index == p.index


def test_clean_paragraph_fallback_when_empty() -> None:
    """If cleaning strips all content, fallback to raw text to avoid empty paragraph."""
    p = Paragraph(
        paper_id="p1",
        section="sec",
        index=0,
        text="BIBREF0",
    )
    cleaned_p = clean_paragraph(p)
    assert cleaned_p.text == "BIBREF0"
    assert cleaned_p.raw_text == "BIBREF0"


def test_clean_paragraph_idempotence() -> None:
    """clean_paragraph(clean_paragraph(p)) preserves raw_text and cleaned text."""
    p = Paragraph(
        paper_id="test",
        section="sec",
        index=0,
        text="Sample with BIBREF1 and TABREF2.",
    )
    c1 = clean_paragraph(p)
    c2 = clean_paragraph(c1)
    assert c2.text == c1.text
    assert c2.raw_text == p.text


def test_clean_paper_preserves_raw_texts() -> None:
    """clean_paper cleans all paragraphs while retaining paper metadata and raw_text."""
    paper = Paper(
        paper_id="test-paper",
        title="Paper with BIBREF0 in Title",
        abstract="Abstract with FIGREF1.",
        paragraphs=[
            Paragraph(paper_id="test-paper", section="S1", index=0, text="Para 0 BIBREF0."),
            Paragraph(paper_id="test-paper", section="S2", index=1, text="Para 1 TABREF1."),
        ],
    )
    cleaned_paper = clean_paper(paper)
    assert cleaned_paper.paper_id == "test-paper"
    assert cleaned_paper.title == "Paper with in Title"
    assert cleaned_paper.abstract == "Abstract with."
    assert len(cleaned_paper.paragraphs) == 2
    assert cleaned_paper.paragraphs[0].text == "Para 0."
    assert cleaned_paper.paragraphs[0].raw_text == "Para 0 BIBREF0."
    assert cleaned_paper.paragraphs[1].text == "Para 1."
    assert cleaned_paper.paragraphs[1].raw_text == "Para 1 TABREF1."


def test_clean_evidence_strips_float_selected() -> None:
    """clean_evidence cleans ground truth evidence strings with FLOAT SELECTED."""
    raw_ev = "FLOAT SELECTED: Table 3: Performance of various models on the ACP test set."
    cleaned_ev = clean_evidence(raw_ev)
    assert cleaned_ev == "Table 3: Performance of various models on the ACP test set."
    assert clean_evidence(cleaned_ev) == cleaned_ev


def test_loader_works_with_cleaning() -> None:
    """Verify papers loaded from fixture can be cleaned cleanly."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    assert len(papers) == 3
    cleaned_papers = [clean_paper(p) for p in papers]
    assert len(cleaned_papers) == 3
    for cp in cleaned_papers:
        assert all(isinstance(p.raw_text, str) and len(p.raw_text) > 0 for p in cp.paragraphs)
        assert all(isinstance(p.text, str) and len(p.text) > 0 for p in cp.paragraphs)
