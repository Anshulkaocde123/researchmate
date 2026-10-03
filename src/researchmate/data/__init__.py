"""Data ingestion, typed models, text cleaning, and evaluation sampling for QASPER."""

from researchmate.data.cleaning import (
    clean_evidence,
    clean_paper,
    clean_paragraph,
    clean_text,
)
from researchmate.data.loader import load_papers, load_split
from researchmate.data.models import Answer, Paper, Paragraph, QAExample
from researchmate.data.sampler import (
    categorize_qa,
    create_and_save_eval_set,
    sample_stratified_qa,
)

__all__ = [
    "Answer",
    "Paper",
    "Paragraph",
    "QAExample",
    "categorize_qa",
    "clean_evidence",
    "clean_paper",
    "clean_paragraph",
    "clean_text",
    "create_and_save_eval_set",
    "load_papers",
    "load_split",
    "sample_stratified_qa",
]
