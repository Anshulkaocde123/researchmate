"""End-to-end RAG pipeline for citation-grounded research paper QA."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from researchmate.generation.citations import CitationResult, extract_citations
from researchmate.generation.generator import GeminiGenerator
from researchmate.generation.prompt import build_rag_prompt
from researchmate.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Structured response from the complete RAG pipeline."""

    question: str
    answer: str
    paper_id: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    citations: CitationResult
    latency_retrieval_sec: float = 0.0
    latency_generation_sec: float = 0.0
    total_latency_sec: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class RAGPipeline:
    """Coordinates hybrid retrieval, prompt grounding, Gemini generation, and citation extraction."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: GeminiGenerator | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or GeminiGenerator()

    def answer_question(
        self,
        question: str,
        paper_id: str | None = None,
        top_k: int = 5,
        rerank: bool = True,
    ) -> RAGResponse:
        """Answer a user question grounded in retrieved paper chunks.

        Args:
            question: Natural language question.
            paper_id: Optional paper ID filter.
            top_k: Number of retrieved chunks to include in the context prompt.
            rerank: Whether to apply cross-encoder reranking.

        Returns:
            RAGResponse object with answer, citations, and latency metrics.
        """
        start_total = time.perf_counter()

        # 1. Retrieval
        t0 = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            paper_id=paper_id,
            top_k=top_k,
            rerank=rerank,
        )
        latency_retrieval = time.perf_counter() - t0

        paper_title = None
        if retrieved_chunks:
            paper_title = retrieved_chunks[0].get("metadata", {}).get("title")

        # 2. Prompt Construction
        system_instruction, user_prompt = build_rag_prompt(
            question=question,
            context_chunks=retrieved_chunks,
            paper_title=paper_title,
        )

        # 3. Generation
        t1 = time.perf_counter()
        gen_result = self.generator.generate(
            user_prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.0,
        )
        latency_generation = time.perf_counter() - t1

        # 4. Citation Extraction & Validation
        citations = extract_citations(
            answer=gen_result.text,
            context_chunks=retrieved_chunks,
        )

        total_latency = time.perf_counter() - start_total

        return RAGResponse(
            question=question,
            answer=gen_result.text,
            paper_id=paper_id,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            latency_retrieval_sec=latency_retrieval,
            latency_generation_sec=latency_generation,
            total_latency_sec=total_latency,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
        )
