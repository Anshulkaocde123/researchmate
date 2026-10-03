"""Unit tests for data models and QASPER loader.

All tests run against tests/fixtures/mini_qasper.json (3 trimmed real papers).
No network calls; no dependency on data/raw/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from researchmate.data.loader import load_papers
from researchmate.data.models import Answer, Paper, Paragraph, QAExample

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mini_qasper.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded() -> tuple[list[Paper], list[QAExample]]:
    """Load the mini fixture once for the whole module."""
    return load_papers(FIXTURE)


@pytest.fixture(scope="module")
def papers(loaded: tuple[list[Paper], list[QAExample]]) -> list[Paper]:
    return loaded[0]


@pytest.fixture(scope="module")
def qa_examples(loaded: tuple[list[Paper], list[QAExample]]) -> list[QAExample]:
    return loaded[1]


@pytest.fixture(scope="module")
def paper_ids(papers: list[Paper]) -> set[str]:
    return {p.paper_id for p in papers}


# ---------------------------------------------------------------------------
# Fixture loads
# ---------------------------------------------------------------------------


class TestFixtureLoads:
    def test_fixture_file_exists(self) -> None:
        assert FIXTURE.exists(), f"Fixture not found: {FIXTURE}"

    def test_returns_three_papers(self, papers: list[Paper]) -> None:
        assert len(papers) == 3

    def test_returns_qa_examples(self, qa_examples: list[QAExample]) -> None:
        assert len(qa_examples) > 0

    def test_papers_are_paper_instances(self, papers: list[Paper]) -> None:
        assert all(isinstance(p, Paper) for p in papers)

    def test_qa_examples_are_qaexample_instances(self, qa_examples: list[QAExample]) -> None:
        assert all(isinstance(q, QAExample) for q in qa_examples)


# ---------------------------------------------------------------------------
# Paper structure
# ---------------------------------------------------------------------------


class TestPaperStructure:
    def test_paper_ids_are_strings(self, papers: list[Paper]) -> None:
        assert all(isinstance(p.paper_id, str) and p.paper_id for p in papers)

    def test_titles_are_non_empty(self, papers: list[Paper]) -> None:
        assert all(isinstance(p.title, str) and p.title.strip() for p in papers)

    def test_paragraphs_are_non_empty(self, papers: list[Paper]) -> None:
        """Every paragraph must have non-empty text (loader should skip blank paragraphs)."""
        for paper in papers:
            for para in paper.paragraphs:
                assert para.text.strip(), f"Empty paragraph {para.index} in paper {paper.paper_id}"

    def test_every_paper_has_paragraphs(self, papers: list[Paper]) -> None:
        assert all(len(p.paragraphs) > 0 for p in papers)

    def test_paragraph_indices_are_sequential(self, papers: list[Paper]) -> None:
        for paper in papers:
            indices = [para.index for para in paper.paragraphs]
            assert indices == list(range(len(indices))), (
                f"Non-sequential indices in paper {paper.paper_id}: {indices}"
            )

    def test_paragraph_paper_ids_match(self, papers: list[Paper]) -> None:
        for paper in papers:
            assert all(para.paper_id == paper.paper_id for para in paper.paragraphs)

    def test_section_names_are_strings(self, papers: list[Paper]) -> None:
        for paper in papers:
            assert all(isinstance(para.section, str) for para in paper.paragraphs)

    def test_paragraphs_are_paragraph_instances(self, papers: list[Paper]) -> None:
        for paper in papers:
            assert all(isinstance(p, Paragraph) for p in paper.paragraphs)


# ---------------------------------------------------------------------------
# QA example structure
# ---------------------------------------------------------------------------


class TestQAExampleStructure:
    def test_every_qa_references_existing_paper(
        self,
        qa_examples: list[QAExample],
        paper_ids: set[str],
    ) -> None:
        for qa in qa_examples:
            assert qa.paper_id in paper_ids, (
                f"QA {qa.question_id!r} references unknown paper {qa.paper_id!r}"
            )

    def test_question_ids_are_non_empty(self, qa_examples: list[QAExample]) -> None:
        assert all(isinstance(q.question_id, str) and q.question_id for q in qa_examples)

    def test_questions_are_non_empty(self, qa_examples: list[QAExample]) -> None:
        assert all(isinstance(q.question, str) and q.question.strip() for q in qa_examples)

    def test_answers_are_lists(self, qa_examples: list[QAExample]) -> None:
        assert all(isinstance(q.answers, list) for q in qa_examples)

    def test_each_qa_has_at_least_one_answer(self, qa_examples: list[QAExample]) -> None:
        assert all(len(q.answers) >= 1 for q in qa_examples)


# ---------------------------------------------------------------------------
# Answer types present in fixture
# ---------------------------------------------------------------------------


class TestAnswerTypes:
    def test_fixture_has_unanswerable_question(self, qa_examples: list[QAExample]) -> None:
        """Fixture paper 1705.09665 must contribute at least one unanswerable answer."""
        has_unans = any(ans.unanswerable for qa in qa_examples for ans in qa.answers)
        assert has_unans, "No unanswerable answer found in fixture"

    def test_fixture_has_yes_no_question(self, qa_examples: list[QAExample]) -> None:
        """Fixture paper 1705.09665 or 2003.07723 must have a yes/no answer."""
        has_yesno = any(isinstance(ans.yes_no, bool) for qa in qa_examples for ans in qa.answers)
        assert has_yesno, "No yes/no answer found in fixture"


# ---------------------------------------------------------------------------
# Evidence field
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_evidence_is_always_list(self, qa_examples: list[QAExample]) -> None:
        for qa in qa_examples:
            for ans in qa.answers:
                assert isinstance(ans.evidence, list), (
                    f"Evidence is {type(ans.evidence).__name__} in {qa.question_id!r}"
                )

    def test_evidence_entries_are_strings(self, qa_examples: list[QAExample]) -> None:
        for qa in qa_examples:
            for ans in qa.answers:
                assert all(isinstance(e, str) for e in ans.evidence), (
                    f"Non-string evidence in {qa.question_id!r}"
                )

    def test_evidence_entries_non_empty(self, qa_examples: list[QAExample]) -> None:
        """Evidence strings must not be blank (loader strips empties)."""
        for qa in qa_examples:
            for ans in qa.answers:
                assert all(e.strip() for e in ans.evidence), (
                    f"Blank evidence entry in {qa.question_id!r}"
                )


# ---------------------------------------------------------------------------
# Answer model — unit tests (no fixture needed)
# ---------------------------------------------------------------------------


class TestAnswerModel:
    def test_defaults(self) -> None:
        a = Answer()
        assert a.unanswerable is False
        assert a.yes_no is None
        assert a.extractive_spans == []
        assert a.free_form == ""
        assert a.evidence == []

    def test_yes_no_true(self) -> None:
        a = Answer(yes_no=True)
        assert a.yes_no is True

    def test_yes_no_false(self) -> None:
        a = Answer(yes_no=False)
        assert a.yes_no is False

    def test_evidence_type_validation(self) -> None:
        with pytest.raises(ValidationError):
            Answer(evidence=[1, 2, 3])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Paragraph model — unit tests
# ---------------------------------------------------------------------------


class TestParagraphModel:
    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Paragraph(paper_id="abc", section="Intro", index=0, text="   ")

    def test_valid_paragraph(self) -> None:
        p = Paragraph(paper_id="abc", section="Intro", index=0, text="Hello world.")
        assert p.text == "Hello world."
