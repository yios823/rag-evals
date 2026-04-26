"""Smoke tests for chunking. Heavy ML deps (sentence-transformers, faiss) are exercised
in the CI eval workflow rather than here, to keep unit tests fast."""

from __future__ import annotations

from rag_evals.ingest import chunk_document


def test_chunking_returns_chunks_with_offsets():
    text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = chunk_document("doc1", text, size=50, overlap=10)
    assert len(chunks) > 1, "expected multiple chunks for a long text"
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.char_end > c.char_start for c in chunks)
    assert chunks[0].chunk_id == "doc1:0"


def test_chunking_handles_empty_text():
    assert chunk_document("doc1", "", size=50, overlap=10) == []


def test_chunking_short_text_single_chunk():
    chunks = chunk_document("doc1", "Just a short sentence.", size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Just a short sentence."
