"""Interactive Gradio Web UI for ResearchMate Citation-Grounded Paper Assistant."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr

from researchmate.config import settings
from researchmate.generation.citations import extract_citations
from researchmate.generation.generator import GeminiGenerator
from researchmate.generation.prompt import build_rag_prompt
from researchmate.indexing.bm25_index import BM25Index
from researchmate.indexing.embedder import FastEmbedder
from researchmate.indexing.vector_store import VectorStore
from researchmate.retrieval.hybrid import HybridRetriever
from researchmate.retrieval.reranker import FastReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_sample_papers(
    eval_path: Path = Path("data/eval/eval_set.jsonl"),
    max_papers: int = 25,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load sample paper titles and their questions from evaluation set."""
    paper_labels: dict[str, str] = {}
    paper_questions: dict[str, list[str]] = defaultdict(list)

    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                pid = item["paper_id"]
                title = item.get("title", pid)
                label = f"{pid} — {title}"
                paper_labels[label] = pid
                if len(paper_questions[label]) < 4:
                    paper_questions[label].append(item["question"])
                if len(paper_labels) >= max_papers:
                    break

    if not paper_labels:
        # Fallback default
        paper_labels["1703.07476 — Topic Identification for Speech without ASR"] = "1703.07476"
        paper_questions["1703.07476 — Topic Identification for Speech without ASR"] = [
            "What datasets are used to assess the performance of the system?",
            "What acoustic features are extracted from the speech signal?",
        ]

    return paper_labels, paper_questions


class UIController:
    """Manages retriever and generator state for the web application."""

    def __init__(self) -> None:
        log.info("Initializing UIController retriever components...")
        bm25_path = settings.index_dir / "bm25.pkl.gz"
        self.bm25_index = BM25Index.load(bm25_path) if bm25_path.exists() else None
        self.vector_store = VectorStore()
        self.embedder = FastEmbedder()
        self.reranker = FastReranker()

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            bm25_index=self.bm25_index,
            reranker=self.reranker,
        )

    def answer_query(
        self,
        paper_choice: str,
        question: str,
        top_k: int,
        rerank: bool,
        mode: str,
        api_key_override: str,
    ) -> tuple[str, str, str]:
        """Execute RAG pipeline and format results for Gradio."""
        if not question or not question.strip():
            return "Please enter a question.", "", ""

        paper_id = (
            paper_choice.split(" — ")[0].strip() if " — " in paper_choice else paper_choice.strip()
        )

        start_time = time.perf_counter()

        # 1. Retrieval
        t_ret_start = time.perf_counter()
        if mode == "Sparse (BM25)":
            retrieved = self.retriever.retrieve_sparse(query=question, paper_id=paper_id, k=top_k)
        elif mode == "Dense (Chroma)":
            retrieved = self.retriever.retrieve_dense(query=question, paper_id=paper_id, k=top_k)
        else:  # Hybrid
            retrieved = self.retriever.retrieve(
                query=question,
                paper_id=paper_id,
                top_k=top_k,
                rerank=rerank,
            )
        t_retrieval = time.perf_counter() - t_ret_start

        paper_title = retrieved[0].get("metadata", {}).get("title") if retrieved else None
        sys_prompt, user_prompt = build_rag_prompt(
            question=question,
            context_chunks=retrieved,
            paper_title=paper_title,
        )

        # 2. Check API Key
        effective_key = (
            api_key_override.strip() or os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
        )

        t_gen = 0.0
        prompt_tokens = None
        completion_tokens = None

        if effective_key:
            try:
                generator = GeminiGenerator(api_key=effective_key)
                t_gen_start = time.perf_counter()
                gen_result = generator.generate(
                    user_prompt=user_prompt,
                    system_instruction=sys_prompt,
                    temperature=0.0,
                )
                t_gen = time.perf_counter() - t_gen_start
                raw_answer = gen_result.text
                prompt_tokens = gen_result.prompt_tokens
                completion_tokens = gen_result.completion_tokens
            except Exception as e:
                raw_answer = (
                    f"⚠️ **Error calling Gemini API:** {e}\n\n*Review retrieved passages below.*"
                )
        else:
            raw_answer = (
                "⚠️ **Gemini API Key Not Configured:**\n\n"
                "To generate a grounded synthetic answer, enter your Gemini API key in the **Settings** panel on the left, "
                "or set `GEMINI_API_KEY` in your `.env` file.\n\n"
                "**Retrieved Evidence Passages** have been retrieved and ranked below for your inspection."
            )

        total_time = time.perf_counter() - start_time

        # 3. Citation Extraction
        citations = extract_citations(raw_answer, retrieved)

        # Format Answer with Badge
        status_badge = (
            "<span style='background:#fef2f2;color:#991b1b;padding:3px 8px;border-radius:4px;font-weight:600;'>❌ Unanswerable from Paper</span>"
            if citations.is_unanswerable
            else "<span style='background:#f0fdf4;color:#166534;padding:3px 8px;border-radius:4px;font-weight:600;'>✅ Grounded Answer</span>"
        )

        formatted_answer = f"{status_badge}\n\n{raw_answer}"

        # Format Stats Bar
        token_info = (
            f" • **Tokens:** {prompt_tokens or 0} prompt + {completion_tokens or 0} output"
            if prompt_tokens
            else ""
        )
        stats_md = (
            f"⏱️ **Total Latency:** {total_time:.2f}s "
            f"(Retrieval: {t_retrieval:.2f}s | Generation: {t_gen:.2f}s){token_info} • "
            f"**Citations:** {len(citations.cited_passage_nums)} valid"
        )

        # Format Retrieved Passages Breakdown
        passages_md_list = []
        for idx, chunk in enumerate(retrieved, start=1):
            cid = chunk.get("chunk_id", "")
            meta = chunk.get("metadata", {})
            sec = meta.get("section", "Section")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            is_cited = idx in citations.cited_passage_nums

            tag = "🌟 **CITED EVIDENCE**" if is_cited else "📄 Candidate Passage"
            border = (
                "border-left: 4px solid #16a34a;" if is_cited else "border-left: 4px solid #cbd5e1;"
            )

            card = (
                f"<div style='margin-bottom:14px;padding:12px 16px;background:#ffffff;border-radius:6px;{border}box-shadow:0 1px 3px rgba(0,0,0,0.05);'>\n"
                f"<div style='font-size:13px;color:#475569;margin-bottom:4px;'>{tag} • <b>[Passage {idx}]</b> • <b>{sec}</b> • Chunk: <code>{cid[:8]}</code> • Score: <code>{score:.3f}</code></div>\n"
                f"<div style='font-size:14px;line-height:1.6;color:#1e293b;'>{text}</div>\n"
                f"</div>"
            )
            passages_md_list.append(card)

        passages_md = "\n".join(passages_md_list) if passages_md_list else "No passages retrieved."

        return formatted_answer, stats_md, passages_md


def create_ui() -> gr.Blocks:
    """Build and return the Gradio UI application."""
    paper_labels, paper_questions = load_sample_papers()
    paper_choices = list(paper_labels.keys())
    default_paper = paper_choices[0]
    default_questions = paper_questions.get(default_paper, [""])

    controller = UIController()

    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-box {
        margin-bottom: 20px;
        padding: 16px 20px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 8px;
        color: white;
    }
    .header-box h1 {
        margin: 0;
        font-size: 24px;
        color: #f8fafc;
    }
    .header-box p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #94a3b8;
    }
    """

    with gr.Blocks(
        title="ResearchMate — Citation-Grounded Paper Assistant", css=custom_css
    ) as demo:
        gr.HTML(
            """
            <div class="header-box">
                <h1>🔬 ResearchMate: Citation-Grounded RAG Assistant</h1>
                <p>Ask scientific questions over NLP papers. Answers are strictly grounded with verbatim bracketed passage citations.</p>
            </div>
            """
        )

        with gr.Row():
            # ── Left Column: Controls & Query ────────────────────────────────
            with gr.Column(scale=4):
                paper_dropdown = gr.Dropdown(
                    label="Select Research Paper",
                    choices=paper_choices,
                    value=default_paper,
                    interactive=True,
                )

                sample_radio = gr.Radio(
                    label="Sample Paper Questions (Click to Populate)",
                    choices=default_questions,
                    value=default_questions[0] if default_questions else None,
                    interactive=True,
                )

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question about this paper...",
                    value=default_questions[0] if default_questions else "",
                    lines=3,
                )

                with gr.Accordion("⚙️ Retrieval & Model Settings", open=False):
                    mode_radio = gr.Radio(
                        label="Retrieval Mode",
                        choices=["Hybrid (Chroma + BM25 + RRF)", "Sparse (BM25)", "Dense (Chroma)"],
                        value="Hybrid (Chroma + BM25 + RRF)",
                    )
                    top_k_slider = gr.Slider(
                        label="Top-k Context Passages",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                    )
                    rerank_checkbox = gr.Checkbox(
                        label="Apply Cross-Encoder Reranking (bge-reranker-base)",
                        value=True,
                    )
                    api_key_box = gr.Textbox(
                        label="Gemini API Key (Optional Override)",
                        placeholder="AIzaSy...",
                        type="password",
                    )

                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask Question", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

            # ── Right Column: Grounded Answer & Evidences ─────────────────────
            with gr.Column(scale=6):
                stats_output = gr.Markdown(value="Ready.")
                answer_output = gr.Markdown(
                    label="Grounded Answer",
                    value="*Submit a question to see the citation-grounded response.*",
                )
                with gr.Accordion("📚 Retrieved Context Passages & Evidence", open=True):
                    passages_output = gr.HTML(value="*Retrieved passages will appear here.*")

        # ── Interactions ─────────────────────────────────────────────────────
        def on_paper_change(selected_paper: str) -> tuple[Any, str]:
            qs = paper_questions.get(selected_paper, [])
            default_q = qs[0] if qs else ""
            return gr.update(choices=qs, value=default_q), default_q

        paper_dropdown.change(
            fn=on_paper_change,
            inputs=[paper_dropdown],
            outputs=[sample_radio, question_input],
        )

        sample_radio.change(
            fn=lambda q: q,
            inputs=[sample_radio],
            outputs=[question_input],
        )

        submit_btn.click(
            fn=controller.answer_query,
            inputs=[
                paper_dropdown,
                question_input,
                top_k_slider,
                rerank_checkbox,
                mode_radio,
                api_key_box,
            ],
            outputs=[answer_output, stats_output, passages_output],
        )

        clear_btn.click(
            fn=lambda: ("", "Ready.", "*Retrieved passages will appear here.*"),
            outputs=[answer_output, stats_output, passages_output],
        )

    return demo


def main() -> None:
    """CLI Entrypoint to run Gradio app."""
    demo = create_ui()
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)  # noqa: S104


if __name__ == "__main__":
    main()
