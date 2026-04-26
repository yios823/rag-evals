"""Ingest a markdown corpus into a hybrid index (FAISS dense + BM25)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import tiktoken

from .config import CONFIG
from .observability import trace


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int


def _tokenize(text: str) -> list[int]:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


def _detokenize(tokens: list[int]) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


def chunk_document(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Token-aware chunking with overlap. Returns chunks with char offsets back to source."""
    tokens = _tokenize(text)
    if len(tokens) == 0:
        return []
    out: list[Chunk] = []
    step = size - overlap
    cursor_chars = 0
    for i in range(0, len(tokens), step):
        window = tokens[i : i + size]
        if not window:
            break
        chunk_text = _detokenize(window)
        # locate this chunk in original text starting at the running cursor
        idx = text.find(chunk_text[: min(40, len(chunk_text))], cursor_chars)
        if idx == -1:
            idx = cursor_chars
        end = idx + len(chunk_text)
        out.append(
            Chunk(
                chunk_id=f"{doc_id}:{i}",
                doc_id=doc_id,
                text=chunk_text,
                char_start=idx,
                char_end=end,
            )
        )
        cursor_chars = idx + max(1, len(chunk_text) - 200)  # advance, but allow overlap
        if i + size >= len(tokens):
            break
    return out


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("**/*.md")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_document(doc_id, text, CONFIG.chunk_size_tokens, CONFIG.chunk_overlap_tokens)
        )
    return chunks


def build_indices(chunks: list[Chunk], out_dir: Path) -> None:
    """Build dense FAISS index + BM25 index, persisted to out_dir."""
    from sentence_transformers import SentenceTransformer
    import faiss
    from rank_bm25 import BM25Okapi

    out_dir.mkdir(parents=True, exist_ok=True)

    with trace("ingest.embed", n_chunks=len(chunks), model=CONFIG.embedding_model):
        model = SentenceTransformer(CONFIG.embedding_model)
        embeddings = model.encode(
            [c.text for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    with trace("ingest.faiss", dim=embeddings.shape[1]):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "dense.faiss"))

    with trace("ingest.bm25"):
        tokenized = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        with open(out_dir / "bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)

    # persist chunks + config (so traces and replays line up)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                    }
                )
                + "\n"
            )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG.as_dict(), f, indent=2)


@click.command("rag-ingest")
@click.option("--corpus", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
def cli(corpus: Path, out: Path) -> None:
    chunks = load_corpus(corpus)
    if not chunks:
        raise click.ClickException(f"No markdown files found under {corpus}")
    build_indices(chunks, out)
    click.echo(f"✓ ingested {len(chunks)} chunks from {corpus} → {out}")


if __name__ == "__main__":
    cli()
