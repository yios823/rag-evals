"""Golden-dataset schema. One labeled query per JSONL line."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    qid: str = Field(..., description="Stable query ID, used as a key in results.")
    question: str
    expected_doc_ids: list[str] = Field(
        ...,
        description="Doc IDs that, if returned in retrieval, count as a relevant hit.",
    )
    reference_answer: str = Field(
        ...,
        description="Human-written reference answer; used by the LLM-judge faithfulness scorer.",
    )
    must_refuse: bool = Field(
        default=False,
        description="True if this query is intentionally unanswerable from the corpus.",
    )
