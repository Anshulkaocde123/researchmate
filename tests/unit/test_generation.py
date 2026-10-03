"""Unit tests for generation, prompt formatting, citation extraction, and RAG pipeline.

All tests run without hitting the network or calling external LLM APIs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from researchmate.generation.citations import extract_citations
from researchmate.generation.generator import GeminiGenerator
from researchmate.generation.prompt import (
    SYSTEM_PROMPT,
    UNANSWERABLE_PHRASE,
    build_rag_prompt,
    format_context_passages,
)
from researchmate.generation.rag_pipeline import RAGPipeline


def test_format_context_passages() -> None:
    """format_context_passages produces structured passage blocks."""
    # Empty case
    assert format_context_passages([]) == "No relevant passages found."

    chunks = [
        {
            "chunk_id": "c1",
            "text": "This paper presents BERT.",
            "metadata": {"title": "BERT Paper", "section": "1. Introduction"},
        },
        {
            "chunk_id": "c2",
            "text": "We evaluate on GLUE benchmark.",
            "metadata": {"title": "BERT Paper", "section": "5. Experiments"},
        },
    ]

    formatted = format_context_passages(chunks)
    assert "[Passage 1]" in formatted
    assert "[Passage 2]" in formatted
    assert "BERT Paper" in formatted
    assert "5. Experiments" in formatted
    assert "We evaluate on GLUE benchmark." in formatted


def test_build_rag_prompt() -> None:
    """build_rag_prompt constructs system and user prompts adhering to grounding rules."""
    chunks = [
        {
            "chunk_id": "c1",
            "text": "RoBERTa is trained on 160GB of text.",
            "metadata": {"title": "RoBERTa", "section": "Training"},
        }
    ]
    sys_prompt, user_prompt = build_rag_prompt("How much text?", chunks, paper_title="RoBERTa")

    assert sys_prompt == SYSTEM_PROMPT
    assert "Target Paper: RoBERTa" in user_prompt
    assert "[Passage 1]" in user_prompt
    assert "Question: How much text?" in user_prompt


def test_extract_citations() -> None:
    """extract_citations parses citations and maps them accurately to context chunks."""
    chunks = [
        {"chunk_id": "chunk_A", "text": "Text for passage 1", "metadata": {"section": "Intro"}},
        {"chunk_id": "chunk_B", "text": "Text for passage 2", "metadata": {"section": "Methods"}},
    ]

    # Test single citation
    ans1 = "The model was pre-trained using Masked LM [Passage 1]."
    res1 = extract_citations(ans1, chunks)
    assert res1.cited_passage_nums == [1]
    assert res1.cited_chunk_ids == ["chunk_A"]
    assert res1.has_valid_citations is True
    assert res1.is_unanswerable is False

    # Test multiple citations
    ans2 = "The experiments show strong results [Passage 1, 2]."
    res2 = extract_citations(ans2, chunks)
    assert res2.cited_passage_nums == [1, 2]
    assert res2.cited_chunk_ids == ["chunk_A", "chunk_B"]

    # Test unanswerable answer
    ans_unans = f"{UNANSWERABLE_PHRASE}"
    res_unans = extract_citations(ans_unans, chunks)
    assert res_unans.is_unanswerable is True
    assert res_unans.has_valid_citations is True

    # Test out-of-range citation
    ans_invalid = "Fact here [Passage 99]."
    res_invalid = extract_citations(ans_invalid, chunks)
    assert res_invalid.cited_passage_nums == [99]
    assert len(res_invalid.cited_chunk_ids) == 0


def test_gemini_generator_mocked() -> None:
    """GeminiGenerator interacts properly with google-genai client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "BERT uses bidirectional self-attention [Passage 1]."
    mock_response.usage_metadata.prompt_token_count = 150
    mock_response.usage_metadata.candidates_token_count = 25

    mock_client.models.generate_content.return_value = mock_response

    generator = GeminiGenerator(client=mock_client)
    res = generator.generate(
        user_prompt="Explain BERT",
        system_instruction="Grounding instruction",
    )

    assert res.text == "BERT uses bidirectional self-attention [Passage 1]."
    assert res.prompt_tokens == 150
    assert res.completion_tokens == 25
    assert res.model == "gemini-2.5-flash"
    mock_client.models.generate_content.assert_called_once()


def test_rag_pipeline_mocked() -> None:
    """End-to-end RAGPipeline coordinates retrieval and generation."""
    mock_retriever = MagicMock()
    mock_generator = MagicMock()

    mock_retriever.retrieve.return_value = [
        {
            "chunk_id": "c_1",
            "text": "The transformer model achieves 28.4 BLEU.",
            "metadata": {"title": "Attention Is All You Need", "section": "Results"},
            "score": 0.95,
        }
    ]

    from researchmate.generation.generator import GenerationResult

    mock_generator.generate.return_value = GenerationResult(
        text="The model scored 28.4 BLEU [Passage 1].",
        model="gemini-2.5-flash",
        prompt_tokens=80,
        completion_tokens=15,
    )

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.answer_question("What is the BLEU score?", paper_id="1706.03762")

    assert response.question == "What is the BLEU score?"
    assert "28.4 BLEU" in response.answer
    assert response.citations.cited_passage_nums == [1]
    assert response.citations.cited_chunk_ids == ["c_1"]
    assert len(response.retrieved_chunks) == 1
    assert response.total_latency_sec > 0.0
