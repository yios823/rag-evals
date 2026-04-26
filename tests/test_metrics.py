from __future__ import annotations

from rag_evals.eval.metrics import (
    aggregate,
    context_precision,
    hallucination_flag,
    recall_at_k,
    reciprocal_rank,
    refusal_correct,
)


def _record(retrieved_doc_ids, expected, **kw):
    return {
        "expected_doc_ids": expected,
        "retrieved": [{"doc_id": d, "chunk_id": d + ":0", "score": 0.0} for d in retrieved_doc_ids],
        **kw,
    }


def test_recall_at_k_perfect():
    rec = _record(["a", "b", "c"], ["a", "b"])
    assert recall_at_k(rec, k=3) == 1.0


def test_recall_at_k_partial():
    rec = _record(["a", "x", "y"], ["a", "b"])
    assert recall_at_k(rec, k=3) == 0.5


def test_recall_at_k_no_expected_returns_one():
    rec = _record(["a", "b"], [])
    assert recall_at_k(rec, k=3) == 1.0


def test_reciprocal_rank_first_position():
    rec = _record(["a", "x", "y"], ["a"])
    assert reciprocal_rank(rec) == 1.0


def test_reciprocal_rank_third_position():
    rec = _record(["x", "y", "a"], ["a"])
    assert abs(reciprocal_rank(rec) - 1.0 / 3) < 1e-9


def test_reciprocal_rank_no_match():
    rec = _record(["x", "y"], ["a"])
    assert reciprocal_rank(rec) == 0.0


def test_context_precision():
    rec = _record(["a", "b", "x"], ["a", "b"])
    assert abs(context_precision(rec, k=3) - 2.0 / 3) < 1e-9


def test_refusal_correct_match():
    rec = _record([], [], must_refuse=True, refused=True)
    assert refusal_correct(rec) == 1.0


def test_refusal_correct_mismatch():
    rec = _record([], ["a"], must_refuse=False, refused=True)
    assert refusal_correct(rec) == 0.0


def test_hallucination_flag_set():
    rec = _record(["a"], ["a"], generator_error="hallucinated_citation: foo")
    assert hallucination_flag(rec) == 1.0


def test_aggregate_returns_all_keys():
    records = [_record(["a"], ["a"], must_refuse=False, refused=False)]
    out = aggregate(records)
    assert set(out) == {
        "recall@5",
        "mrr@10",
        "ctx_precision@5",
        "hallucination_rate",
        "refusal_correct",
    }
