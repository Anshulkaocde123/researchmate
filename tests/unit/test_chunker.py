"""Unit tests for section-aware paragraph and sentence chunker.

Tests cover:
- Hard token ceiling: no chunk exceeds max_tokens
- Full paragraph coverage: every paragraph is represented in at least one chunk
- Determinism: chunk_ids are completely deterministic across independent runs
- Overlap behavior: consecutive chunks share paragraphs
- Sentence splitting fallback: oversized paragraphs split cleanly on sentences
- Empty section skipping: sections with no valid paragraphs are omitted
- Text separation: embed_text has Title > Section prefix while text remains clean
- Real fixture test: chunking all 3 papers in mini_qasper.json fixture
"""

from __future__ import annotations

from pathlib import Path

from researchmate.chunking.chunker import (
    Chunk,
    chunk_paper,
    chunk_section,
)
from researchmate.data.loader import load_papers
from researchmate.data.models import Paper, Paragraph

# =========================================================================
# 1. Determinism and IDs
# =========================================================================


def test_chunk_ids_are_deterministic() -> None:
    """Chunk IDs and contents must be 100% deterministic across multiple runs."""
    paper = Paper(
        paper_id="2002.06424",
        title="Affective Events Paper",
        paragraphs=[
            Paragraph(
                paper_id="2002.06424", section="Introduction", index=0, text="First paragraph."
            ),
            Paragraph(
                paper_id="2002.06424", section="Introduction", index=1, text="Second paragraph."
            ),
            Paragraph(paper_id="2002.06424", section="Methods", index=2, text="Third paragraph."),
        ],
    )

    chunks_run1 = chunk_paper(paper, max_tokens=100, overlap_paragraphs=1)
    chunks_run2 = chunk_paper(paper, max_tokens=100, overlap_paragraphs=1)

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.embed_text == c2.embed_text


# =========================================================================
# 2. Section Boundaries and Skipping
# =========================================================================


def test_chunk_never_crosses_section_boundary() -> None:
    """Chunks must strictly belong to a single section."""
    paper = Paper(
        paper_id="2002.06424",
        title="Test Paper",
        paragraphs=[
            Paragraph(paper_id="2002.06424", section="Intro", index=0, text="Intro text."),
            Paragraph(paper_id="2002.06424", section="Methods", index=1, text="Methods text."),
            Paragraph(paper_id="2002.06424", section="Results", index=2, text="Results text."),
        ],
    )

    chunks = chunk_paper(paper, max_tokens=400, overlap_paragraphs=1)
    # Even if max_tokens is 400 and all paragraphs fit, there must be 3 separate chunks
    assert len(chunks) == 3
    assert [c.section for c in chunks] == ["Intro", "Methods", "Results"]


def test_empty_sections_are_skipped() -> None:
    """Empty sections or empty paragraph lists produce no chunks."""
    assert chunk_section("p1", "Title", "", []) == []
    assert chunk_section("p1", "Title", "Valid Section", []) == []


# =========================================================================
# 3. Token Ceiling and Coverage
# =========================================================================


def test_no_chunk_exceeds_max_tokens() -> None:
    """No chunk may have n_tokens greater than max_tokens."""
    max_tokens = 50
    paper = Paper(
        paper_id="p1",
        title="Paper Title",
        paragraphs=[
            Paragraph(
                paper_id="p1",
                section="Section 1",
                index=i,
                text=f"Paragraph {i} has several words to test token ceiling and packing.",
            )
            for i in range(15)
        ],
    )

    chunks = chunk_paper(paper, max_tokens=max_tokens, overlap_paragraphs=1)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.n_tokens <= max_tokens


def test_every_paragraph_is_covered_at_least_once() -> None:
    """Every paragraph in the paper must appear in at least one chunk."""
    n_paras = 10
    paper = Paper(
        paper_id="p1",
        title="Paper Title",
        paragraphs=[
            Paragraph(
                paper_id="p1",
                section="Section 1",
                index=i,
                text=f"Paragraph index {i} text content for testing complete coverage.",
            )
            for i in range(n_paras)
        ],
    )

    chunks = chunk_paper(paper, max_tokens=40, overlap_paragraphs=1)
    covered_indices: set[int] = set()
    for chunk in chunks:
        for idx in range(chunk.para_start, chunk.para_end + 1):
            covered_indices.add(idx)

    assert covered_indices == set(range(n_paras))


# =========================================================================
# 4. Overlap and Sentence Splitting
# =========================================================================


def test_overlap_paragraphs_work() -> None:
    """Consecutive chunks in the same section overlap by specified paragraph count."""
    paper = Paper(
        paper_id="p1",
        title="Paper Title",
        paragraphs=[
            Paragraph(
                paper_id="p1",
                section="Section 1",
                index=i,
                text=f"Paragraph {i} contains exactly eight words here.",
            )
            for i in range(6)
        ],
    )

    # 8 words per para. If max_tokens = 20, 2 paras fit per chunk (16 tokens)
    chunks = chunk_paper(
        paper,
        max_tokens=20,
        overlap_paragraphs=1,
        token_counter=lambda s: len(s.split()),
    )

    assert len(chunks) >= 4
    # Check that chunk 0 has para 0-1, chunk 1 has para 1-2 (overlapping para 1)
    assert chunks[0].para_start == 0 and chunks[0].para_end == 1
    assert chunks[1].para_start == 1 and chunks[1].para_end == 2


def test_very_long_paragraph_splits_on_sentences() -> None:
    """A paragraph exceeding max_tokens alone splits into sentence-level chunks."""
    long_sentences = [f"This is sentence number {i} with some words." for i in range(10)]
    long_para_text = " ".join(long_sentences)

    paper = Paper(
        paper_id="p1",
        title="Long Para Paper",
        paragraphs=[
            Paragraph(paper_id="p1", section="Deep Dive", index=0, text=long_para_text),
        ],
    )

    # max_tokens = 30 -> will need multiple sentence chunks
    chunks = chunk_paper(
        paper,
        max_tokens=30,
        token_counter=lambda s: len(s.split()),
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.para_start == 0
        assert chunk.para_end == 0
        assert chunk.n_tokens <= 30
        assert len(chunk.text) > 0


# =========================================================================
# 5. Embed Text vs Display Text
# =========================================================================


def test_embed_text_has_prefix_and_text_does_not() -> None:
    """embed_text contains Title > Section prefix; text contains pure display text."""
    paper = Paper(
        paper_id="2002.06424",
        title="Minimally Supervised Affective Events",
        paragraphs=[
            Paragraph(
                paper_id="2002.06424",
                section="Introduction",
                index=0,
                text="Affective events are events that affect people.",
            ),
        ],
    )

    chunks = chunk_paper(paper, max_tokens=100)
    assert len(chunks) == 1
    c: Chunk = chunks[0]

    assert c.text == "Affective events are events that affect people."
    assert "Minimally Supervised Affective Events > Introduction" not in c.text
    assert c.embed_text.startswith("Minimally Supervised Affective Events > Introduction\n\n")
    assert c.embed_text.endswith("Affective events are events that affect people.")


# =========================================================================
# 6. Real Fixture Spot-Check on 3 Papers
# =========================================================================


def test_chunking_on_mini_qasper_fixture() -> None:
    """Spot-check chunking across all 3 papers in tests/fixtures/mini_qasper.json."""
    papers, _ = load_papers(Path("tests/fixtures/mini_qasper.json"))
    assert len(papers) == 3

    for paper in papers:
        chunks = chunk_paper(paper, max_tokens=400, overlap_paragraphs=1)
        assert len(chunks) > 0

        # Every chunk has valid metadata
        for c in chunks:
            assert c.paper_id == paper.paper_id
            assert c.title == paper.title
            assert len(c.section) > 0
            assert len(c.text) > 0
            assert c.n_tokens <= 400
            assert c.embed_text.startswith(f"{paper.title} > {c.section}\n\n")
            assert not c.text.startswith(f"{paper.title} >")

        # Paragraph coverage
        covered = set()
        for c in chunks:
            for idx in range(c.para_start, c.para_end + 1):
                covered.add(idx)
        total_paras = len(paper.paragraphs)
        assert covered == set(range(total_paras))


def test_chunker_empty_paper_and_empty_text() -> None:
    """chunk_paper handles empty paper and default token counter handles empty text."""
    from researchmate.chunking.chunker import default_token_counter, get_token_counter

    assert default_token_counter("") == 0
    empty_paper = Paper(paper_id="empty", title="Empty", paragraphs=[])
    assert chunk_paper(empty_paper) == []

    counter = get_token_counter("non_existent_model")
    assert counter("test string") > 0
