"""Cross-encoder reranker. Runs after hybrid retrieval to improve top-k precision."""

from __future__ import annotations

from .config import CONFIG
from .observability import trace
from .retrieve import Hit


class Reranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(CONFIG.reranker_model)

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        if not hits:
            return []
        k = top_k or CONFIG.rerank_top_k
        with trace("rerank.cross_encoder", query=query, n_in=len(hits), top_k=k) as span:
            pairs = [(query, h.text) for h in hits]
            scores = self.model.predict(pairs).tolist()
            ranked = sorted(zip(hits, scores), key=lambda x: -x[1])[:k]
            out = [Hit(chunk_id=h.chunk_id, doc_id=h.doc_id, text=h.text, score=float(s)) for h, s in ranked]
            span["top1"] = out[0].chunk_id if out else None
            return out
