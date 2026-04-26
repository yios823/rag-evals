"""Hybrid retrieval: dense (FAISS) + BM25 → reciprocal-rank fusion."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CONFIG
from .observability import trace


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class HybridRetriever:
    def __init__(self, index_dir: Path) -> None:
        from sentence_transformers import SentenceTransformer
        import faiss

        self.index_dir = index_dir
        chunks_path = index_dir / "chunks.jsonl"
        self._chunks = [
            json.loads(line)
            for line in chunks_path.read_text().splitlines()
        ]
        self._faiss = faiss.read_index(str(index_dir / "dense.faiss"))
        with open(index_dir / "bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        self._embed = SentenceTransformer(CONFIG.embedding_model)

    def _dense_topn(
        self, query: str, n: int
    ) -> list[tuple[int, float]]:
        n = min(n, self._faiss.ntotal)
        if n == 0:
            return []
        q = self._embed.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, idxs = self._faiss.search(q, n)
        # FAISS returns -1 for empty slots when k > ntotal; we
        # already clamp above, but filter to be safe.
        return [
            (int(i), float(s))
            for i, s in zip(idxs[0].tolist(), scores[0].tolist())
            if i >= 0
        ]

    def _bm25_topn(
        self, query: str, n: int
    ) -> list[tuple[int, float]]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        n = min(n, len(scores))
        if n == 0:
            return []
        top = np.argpartition(scores, -n)[-n:]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def search(
        self, query: str, top_n: int | None = None
    ) -> list[Hit]:
        n = top_n or CONFIG.retrieve_top_n
        with trace("retrieve.hybrid", query=query, top_n=n) as span:
            dense = self._dense_topn(query, n)
            bm25 = self._bm25_topn(query, n)
            fused = self._reciprocal_rank_fusion(
                [dense, bm25], k=CONFIG.rrf_k
            )
            hits = [
                Hit(
                    chunk_id=self._chunks[i]["chunk_id"],
                    doc_id=self._chunks[i]["doc_id"],
                    text=self._chunks[i]["text"],
                    score=score,
                )
                for i, score in fused[:n]
            ]
            span["dense_top1"] = (
                self._chunks[dense[0][0]]["chunk_id"]
                if dense
                else None
            )
            span["bm25_top1"] = (
                self._chunks[bm25[0][0]]["chunk_id"]
                if bm25
                else None
            )
            span["fused_top1"] = hits[0].chunk_id if hits else None
            return hits

    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: list[list[tuple[int, float]]], k: int
    ) -> list[tuple[int, float]]:
        """Standard RRF: score(i) = sum over rankings of
        1 / (k + rank_in_ranking(i))."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _raw_score) in enumerate(ranking):
                scores[idx] = (
                    scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
                )
        return sorted(scores.items(), key=lambda kv: -kv[1])
