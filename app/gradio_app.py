"""Gradio interactive demo, deployable to Hugging Face Spaces.

Spaces config (move to app/README.md when deploying):

  ---
  title: rag-evals demo
  emoji: 🔎
  colorFrom: indigo
  colorTo: purple
  sdk: gradio
  sdk_version: 4.40.0
  app_file: app/gradio_app.py
  ---
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from rag_evals.generate import generate
from rag_evals.rerank import Reranker
from rag_evals.retrieve import HybridRetriever


load_dotenv()
INDEX_DIR = Path(os.getenv("RAG_INDEX", "data/index"))


def _build():
    if not (INDEX_DIR / "chunks.jsonl").exists():
        raise RuntimeError(
            f"No index found at {INDEX_DIR}. Run `make ingest` first."
        )
    return HybridRetriever(INDEX_DIR), Reranker()


RETRIEVER, RERANKER = _build()


def answer(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Ask a question.", ""
    hits = RETRIEVER.search(question)
    reranked = RERANKER.rerank(question, hits)
    try:
        ans = generate(question, reranked)
    except ValueError as e:
        return f"Generator failed validation: {e}", json.dumps(
            [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": h.score} for h in reranked],
            indent=2,
        )
    body = ans.answer
    if ans.refused:
        body = f"(refused) {body}"
    if ans.citation_chunk_ids:
        body += "\n\ncitations: " + ", ".join(ans.citation_chunk_ids)
    chunks_view = json.dumps(
        [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 3)} for h in reranked],
        indent=2,
    )
    return body, chunks_view


with gr.Blocks(title="rag-evals demo") as demo:
    gr.Markdown(
        "# rag-evals demo\n"
        "Hybrid retrieval (dense + BM25) → cross-encoder rerank → Claude with structured output and citation validation. "
        "Code: https://github.com/yios823/rag-evals"
    )
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="Ask something the corpus could answer", lines=2)
    with gr.Row():
        ask = gr.Button("Ask", variant="primary")
    with gr.Row():
        with gr.Column():
            answer_box = gr.Textbox(label="Answer", lines=8)
        with gr.Column():
            chunks_box = gr.Code(label="Top reranked chunks", language="json")
    ask.click(answer, inputs=q, outputs=[answer_box, chunks_box])

if __name__ == "__main__":
    demo.launch()
