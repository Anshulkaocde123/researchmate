"""Integration smoke tests — skipped automatically when GEMINI_API_KEY is absent.

These tests hit the real Google Gemini API and require a valid GEMINI_API_KEY.
"""

from __future__ import annotations

import os

import pytest

from researchmate.generation.generator import GeminiGenerator
from researchmate.generation.rag_pipeline import RAGPipeline


@pytest.mark.integration
def test_integration_marker_is_wired() -> None:
    """Confirms the integration marker and conftest skip logic work."""
    assert os.getenv("GEMINI_API_KEY") is not None


@pytest.mark.integration
def test_real_gemini_generation() -> None:
    """Test actual generation via Gemini API when GEMINI_API_KEY is provided."""
    generator = GeminiGenerator()
    result = generator.generate(
        user_prompt="Reply with exactly the word 'SUCCESS' and nothing else.",
        temperature=0.0,
    )
    assert "SUCCESS" in result.text.upper()


@pytest.mark.integration
def test_real_rag_pipeline_end_to_end() -> None:
    """Test full RAGPipeline with actual Gemini API."""
    pipeline = RAGPipeline()
    response = pipeline.answer_question(
        question="What datasets are used to evaluate topic identification?",
        paper_id="1703.07476",
        top_k=3,
    )
    assert len(response.answer) > 0
    assert response.total_latency_sec > 0.0
