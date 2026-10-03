"""Data ingestion, typed models, and text cleaning for QASPER dataset."""

from researchmate.data.cleaning import (
    clean_evidence,
    clean_paper,
    clean_paragraph,
    clean_text,
)
from researchmate.data.loader import load_papers, load_split
from researchmate.data.models import Answer, Paper, Paragraph, QAExample

__all__ = [
    "Answer",
    "Paper",
    "Paragraph",
    "QAExample",
    "clean_evidence",
    "clean_paper",
    "clean_paragraph",
    "clean_text",
    "load_papers",
    "load_split",
]
