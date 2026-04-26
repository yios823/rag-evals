"""Generator: Anthropic Claude with a Pydantic-validated response.

The response shape is enforced. If the model returns malformed JSON or
cites a chunk ID that wasn't in the retrieved context, this layer
raises rather than passing it downstream.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CONFIG
from .observability import trace
from .retrieve import Hit


class Answer(BaseModel):
    answer: str = Field(
        ...,
        description="Direct answer to the question, grounded in citations.",
    )
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Chunk IDs from the retrieved context that support this "
            "answer. Empty if the question cannot be answered from "
            "context."
        ),
    )
    refused: bool = Field(
        default=False,
        description=(
            "True if the model refused to answer because the context "
            "was insufficient."
        ),
    )


SYSTEM_PROMPT = """You are a careful technical assistant answering \
questions strictly from the provided context.

Rules:
1. Use only information that is in the context. If the context does \
not contain the answer, set `refused` to true and explain briefly.
2. Cite the chunk IDs you used in `citation_chunk_ids`. Only cite IDs \
from the provided context.
3. Be concise. No preamble. No "based on the context" filler.
4. Output must be valid JSON matching the schema. Do not wrap it in \
code fences or commentary.
"""


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk_id}]\n{h.text}" for h in hits)


def _build_user_prompt(question: str, hits: list[Hit]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Context (each chunk has an ID in brackets):\n\n"
        f"{_format_context(hits)}\n\n"
        'Respond with JSON: {"answer": str, '
        '"citation_chunk_ids": [str, ...], "refused": bool}'
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _call_claude(system: str, user: str, max_tokens: int) -> str:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env."
        )
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CONFIG.generator_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _parse(raw: str, valid_chunk_ids: set[str]) -> Answer:
    """Strict JSON parse + Pydantic validation + citation check."""
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Generator returned non-JSON output: {e!s}; "
            f"raw={raw[:200]!r}"
        ) from e
    try:
        ans = Answer(**data)
    except ValidationError as e:
        raise ValueError(
            f"Generator output failed schema validation: {e}"
        ) from e
    bad = [c for c in ans.citation_chunk_ids if c not in valid_chunk_ids]
    if bad:
        raise ValueError(
            f"Generator hallucinated citation IDs not in context: {bad}"
        )
    return ans


def generate(question: str, hits: list[Hit]) -> Answer:
    """Run the generator. Returns a validated Answer, raises on misuse."""
    user = _build_user_prompt(question, hits)
    valid_ids = {h.chunk_id for h in hits}
    with trace(
        "generate.claude",
        model=CONFIG.generator_model,
        question=question,
        n_chunks=len(hits),
    ) as span:
        raw = _call_claude(SYSTEM_PROMPT, user, CONFIG.max_answer_tokens)
        ans = _parse(raw, valid_ids)
        span["refused"] = ans.refused
        span["n_citations"] = len(ans.citation_chunk_ids)
        return ans
