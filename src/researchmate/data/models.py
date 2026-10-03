"""Pydantic models for QASPER papers and QA examples.

All fields reflect the **real v0.3 schema** documented in docs/data_schema.md.
Missing or null values are handled explicitly with defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Paragraph(BaseModel):
    """A single paragraph extracted from a paper's full text.

    ``index`` is the zero-based position within the paper (across all sections).
    """

    paper_id: str = Field(description="arXiv ID of the parent paper.")
    section: str = Field(description="Section heading; '(no section)' when blank.")
    index: int = Field(ge=0, description="Zero-based paragraph index within the paper.")
    text: str = Field(description="Raw paragraph text (may contain BIBREF tokens).")

    @model_validator(mode="after")
    def text_not_empty(self) -> Paragraph:
        if not self.text.strip():
            raise ValueError(f"Paragraph {self.index} in paper {self.paper_id!r} is empty.")
        return self


class Paper(BaseModel):
    """A QASPER paper with its full text flattened to an ordered paragraph list."""

    paper_id: str = Field(description="arXiv ID, e.g. '2002.06424'.")
    title: str = Field(description="Paper title.")
    abstract: str = Field(default="", description="Paper abstract (may be empty).")
    paragraphs: list[Paragraph] = Field(
        default_factory=list,
        description="All body paragraphs in section order, non-empty.",
    )


class Answer(BaseModel):
    """One annotator's answer to a QASPER question.

    At most one of ``extractive_spans``, ``free_form``, ``yes_no`` is set;
    ``unanswerable`` takes priority over all.
    """

    unanswerable: bool = Field(
        default=False,
        description="True when the annotator marked the question as unanswerable.",
    )
    yes_no: bool | None = Field(
        default=None,
        description="True/False for yes-no questions; None otherwise.",
    )
    extractive_spans: list[str] = Field(
        default_factory=list,
        description="Verbatim spans copied from the paper.",
    )
    free_form: str = Field(
        default="",
        description="Paraphrase answer (free-form text).",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting paragraph strings from the paper.",
    )

    @model_validator(mode="after")
    def evidence_is_strings(self) -> Answer:
        if not all(isinstance(e, str) for e in self.evidence):
            raise TypeError("All evidence entries must be strings.")
        return self


class QAExample(BaseModel):
    """One question about a paper, with answers from all annotators."""

    paper_id: str = Field(description="arXiv ID of the paper the question is about.")
    question_id: str = Field(description="Unique question identifier.")
    question: str = Field(description="Question text.")
    answers: list[Answer] = Field(
        default_factory=list,
        description="One Answer per annotator (usually two).",
    )
