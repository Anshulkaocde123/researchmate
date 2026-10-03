"""Generation and RAG pipeline components for citation-grounded scientific QA."""

from researchmate.generation.citations import CitationMatch, CitationResult, extract_citations
from researchmate.generation.generator import GeminiGenerator, GenerationResult
from researchmate.generation.prompt import (
    SYSTEM_PROMPT,
    UNANSWERABLE_PHRASE,
    build_rag_prompt,
    format_context_passages,
)
from researchmate.generation.rag_pipeline import RAGPipeline, RAGResponse

__all__ = [
    "SYSTEM_PROMPT",
    "UNANSWERABLE_PHRASE",
    "CitationMatch",
    "CitationResult",
    "GeminiGenerator",
    "GenerationResult",
    "RAGPipeline",
    "RAGResponse",
    "build_rag_prompt",
    "extract_citations",
    "format_context_passages",
]
