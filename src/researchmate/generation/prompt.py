"""Prompt construction and grounding rules for ResearchMate scientific QA."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """You are ResearchMate, an expert scientific research assistant. Your task is to answer user questions about research papers using ONLY the provided context passages.

Strict Grounding Rules:
1. Base your answer EXCLUSIVELY on the facts provided in the Context Passages below.
2. Do NOT extrapolate, speculate, or introduce external knowledge not explicitly stated in the passages.
3. If the provided passages do NOT contain sufficient information to answer the question, you must respond with:
   "I cannot answer this question based on the provided paper."
4. Every factual claim in your answer MUST cite the source passage using bracketed notation like [Passage 1], [Passage 2].
5. Keep the answer direct, concise, and scientifically accurate. Preserve exact numbers, metrics, model names, and experimental results as stated in the text.
"""

UNANSWERABLE_PHRASE = "I cannot answer this question based on the provided paper."


def format_context_passages(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunk candidates into labeled passages for the prompt.

    Args:
        chunks: List of chunk dictionaries containing text and metadata.

    Returns:
        Formatted string containing labeled passages.
    """
    if not chunks:
        return "No relevant passages found."

    formatted_passages: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        title = meta.get("title", "Untitled")
        section = meta.get("section", "General")
        text = c.get("text", "").strip()

        passage_block = f"--- [Passage {idx}] ---\nPaper: {title}\nSection: {section}\n\n{text}"
        formatted_passages.append(passage_block)

    return "\n\n".join(formatted_passages)


def build_rag_prompt(
    question: str,
    context_chunks: list[dict[str, Any]],
    paper_title: str | None = None,
) -> tuple[str, str]:
    """Construct (system_instruction, user_prompt) tuple for the LLM.

    Args:
        question: User's query or question.
        context_chunks: Retrieved candidate chunks.
        paper_title: Optional title of the paper being queried.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    passages_text = format_context_passages(context_chunks)

    header = f"Target Paper: {paper_title}\n\n" if paper_title else ""
    user_prompt = (
        f"{header}"
        f"Context Passages:\n"
        f"================\n"
        f"{passages_text}\n"
        f"================\n\n"
        f"Question: {question.strip()}\n\n"
        f"Answer the question with specific [Passage X] citations, or state if unanswerable:"
    )

    return SYSTEM_PROMPT, user_prompt
