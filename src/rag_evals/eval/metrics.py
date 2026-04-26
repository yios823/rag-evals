"""Eval metrics. Each takes the per-query record dict and returns a float in [0, 1]."""

from __future__ import annotations


def recall_at_k(record: dict, k: int = 5) -> float:
    """Fraction of expected doc IDs found in the top-k retrieved doc IDs."""
    expected = set(record["expected_doc_ids"])
    if not expected:
        return 1.0
    retrieved = [h["doc_id"] for h in record["retrieved"][:k]]
    hit = len(expected & set(retrieved))
    return hit / len(expected)


def reciprocal_rank(record: dict, k: int = 10) -> float:
    """1 / rank of the first relevant doc among top-k; 0 if none."""
    expected = set(record["expected_doc_ids"])
    for i, h in enumerate(record["retrieved"][:k], start=1):
        if h["doc_id"] in expected:
            return 1.0 / i
    return 0.0


def context_precision(record: dict, k: int = 5) -> float:
    """Fraction of top-k retrieved docs that are in the expected set."""
    expected = set(record["expected_doc_ids"])
    retrieved = [h["doc_id"] for h in record["retrieved"][:k]]
    if not retrieved:
        return 0.0
    hits = sum(1 for d in retrieved if d in expected)
    return hits / len(retrieved)


def hallucination_flag(record: dict) -> float:
    """1.0 if the generator cited a chunk ID that wasn't in retrieved context, else 0.0.

    The generator layer raises on this in production; here we record it for measurement.
    """
    if record.get("generator_error", "").startswith("hallucinated_citation"):
        return 1.0
    return 0.0


def refusal_correct(record: dict) -> float:
    """1.0 if the model's refusal matches whether the question is answerable from corpus."""
    return 1.0 if bool(record.get("refused")) == bool(record.get("must_refuse")) else 0.0


def aggregate(records: list[dict]) -> dict[str, float]:
    """Mean of each metric across all queries."""
    if not records:
        return {}
    n = len(records)
    return {
        "recall@5": sum(recall_at_k(r, 5) for r in records) / n,
        "mrr@10": sum(reciprocal_rank(r, 10) for r in records) / n,
        "ctx_precision@5": sum(context_precision(r, 5) for r in records) / n,
        "hallucination_rate": sum(hallucination_flag(r) for r in records) / n,
        "refusal_correct": sum(refusal_correct(r) for r in records) / n,
    }
