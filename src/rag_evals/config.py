"""Pinned model versions and chunking parameters.

Anything that affects retrieval quality lives here so it shows up in
traces and is diffable in PRs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model: str = "claude-sonnet-4-6"
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    retrieve_top_n: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    max_answer_tokens: int = 800

    def as_dict(self) -> dict:
        return asdict(self)


CONFIG = Config()
