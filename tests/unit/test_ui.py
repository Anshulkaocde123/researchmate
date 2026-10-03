"""Unit tests for ResearchMate Gradio UI controller and components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from researchmate.ui.app import UIController, load_sample_papers


def test_load_sample_papers() -> None:
    """load_sample_papers returns paper labels and questions."""
    labels, questions = load_sample_papers()
    assert len(labels) > 0
    assert len(questions) > 0
    first_label = list(labels.keys())[0]
    assert " — " in first_label
    assert len(questions[first_label]) > 0


@patch("researchmate.ui.app.VectorStore")
@patch("researchmate.ui.app.FastEmbedder")
@patch("researchmate.ui.app.FastReranker")
def test_ui_controller_answer_query(
    mock_reranker: MagicMock,
    mock_embedder: MagicMock,
    mock_vs: MagicMock,
) -> None:
    """UIController executes retrieval and formats response without crashing."""
    controller = UIController()
    controller.retriever = MagicMock()
    controller.retriever.retrieve.return_value = [
        {
            "chunk_id": "c_abc",
            "text": "The transformer architecture relies on self-attention.",
            "score": 0.95,
            "metadata": {"title": "Attention Paper", "section": "3. Architecture"},
        }
    ]

    # Test empty query
    ans_empty, _, _ = controller.answer_query(
        paper_choice="1703.07476 — Title",
        question="",
        top_k=5,
        rerank=True,
        mode="Hybrid",
        api_key_override="",
    )
    assert ans_empty == "Please enter a question."

    # Test retrieval-only mode (no API key)
    ans, stats, passages = controller.answer_query(
        paper_choice="1703.07476 — Title",
        question="What architecture is used?",
        top_k=5,
        rerank=True,
        mode="Hybrid",
        api_key_override="",
    )

    assert "Gemini API Key Not Configured" in ans
    assert "Latency:" in stats
    assert "[Passage 1]" in passages
    assert "3. Architecture" in passages
