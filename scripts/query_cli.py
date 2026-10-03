#!/usr/bin/env python3
"""Interactive Command-Line Interface (CLI) for ResearchMate paper querying.

Usage:
    # Single query:
    python scripts/query_cli.py --paper-id 1703.07476 --question "What datasets are used?"

    # Interactive REPL mode:
    python scripts/query_cli.py --interactive
"""

from __future__ import annotations

import argparse

from researchmate.config import settings
from researchmate.generation.rag_pipeline import RAGPipeline
from researchmate.indexing.bm25_index import BM25Index
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore
from researchmate.retrieval.hybrid import HybridRetriever
from researchmate.retrieval.reranker import FastReranker


def init_pipeline() -> RAGPipeline:
    """Initialize RAG pipeline components."""
    bm25_path = settings.index_dir / "bm25.pkl.gz"
    bm25_index = BM25Index.load(bm25_path) if bm25_path.exists() else None
    vs = VectorStore()
    embedder = FastEmbedder()
    reranker = FastReranker()

    retriever = HybridRetriever(
        vector_store=vs,
        embedder=embedder,
        bm25_index=bm25_index,
        reranker=reranker,
    )
    return RAGPipeline(retriever=retriever)


def run_single_query(
    pipeline: RAGPipeline, question: str, paper_id: str | None = None, top_k: int = 5
) -> None:
    """Execute a query and print formatted output."""
    try:
        response = pipeline.answer_question(question=question, paper_id=paper_id, top_k=top_k)
        print("\n" + "=" * 60)
        print("🤖 GROUNDED ANSWER:")
        print("=" * 60)
        print(response.answer)
        print("=" * 60)

        print(
            f"\n📊 METRICS: Total Latency: {response.total_latency_sec:.2f}s "
            f"(Retrieval: {response.latency_retrieval_sec:.2f}s | Generation: {response.latency_generation_sec:.2f}s)"
        )
        print(f"📌 Valid Citations: {len(response.citations.cited_passage_nums)}")
        retrieved_chunks = response.retrieved_chunks
        cited_nums = response.citations.cited_passage_nums
    except (ValueError, Exception) as err:
        print(f"\n⚠️ Generation notice: {err}")
        print("Retrieving and displaying top candidate passages directly...")
        retrieved_chunks = pipeline.retriever.retrieve(
            query=question, paper_id=paper_id, top_k=top_k
        )
        cited_nums = []

    print("\n📚 RETRIEVED CONTEXT PASSAGES:")
    for idx, c in enumerate(retrieved_chunks, start=1):
        sec = c.get("metadata", {}).get("section", "Section")
        cid = c.get("chunk_id", "")[:8]
        score = c.get("score", 0.0)
        is_cited = idx in cited_nums
        tag = "[CITED] " if is_cited else ""
        print(f"\n  • {tag}[Passage {idx}] ({sec} | Chunk {cid} | Score: {score:.3f}):")
        print(f'    "{c.get("text", "")[:180]}..."')
    print()


def repl_mode(pipeline: RAGPipeline) -> None:
    """Run interactive terminal REPL."""
    print("\n🔬 Welcome to ResearchMate Interactive CLI!")
    print("Type 'exit' or 'quit' to leave.\n")

    current_paper_id = "1703.07476"
    while True:
        try:
            prompt_str = f"[{current_paper_id}] Enter question (or /paper <id>): "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if user_input.startswith("/paper"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    current_paper_id = parts[1].strip()
                    print(f"Switched target paper to: {current_paper_id}")
                else:
                    print(f"Current paper: {current_paper_id}")
                continue

            run_single_query(pipeline, question=user_input, paper_id=current_paper_id)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", type=str, default="1703.07476", help="QASPER Paper ID")
    parser.add_argument("--question", type=str, default=None, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k context passages")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL mode")
    args = parser.parse_args()

    pipeline = init_pipeline()

    if args.interactive or args.question is None:
        repl_mode(pipeline)
    else:
        run_single_query(pipeline, question=args.question, paper_id=args.paper_id, top_k=args.top_k)


if __name__ == "__main__":
    main()
