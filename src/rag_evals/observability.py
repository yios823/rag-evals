"""Tracing.

Uses Langfuse if credentials are set, otherwise writes structured JSON
to stdout. Errors in spans are recorded and re-raised; nothing is
swallowed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


def _stdout_log(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


def _langfuse_client():
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        return Langfuse()
    except ImportError:
        return None


_LANGFUSE = _langfuse_client()


@contextmanager
def trace(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """Emit a span. Yields a dict so callers can add attrs mid-span."""
    start = time.perf_counter()
    span = {"name": name, "attrs": dict(attrs)}
    try:
        yield span
        elapsed_ms = (time.perf_counter() - start) * 1000
        record = {
            "event": "trace",
            "name": name,
            "ms": round(elapsed_ms, 2),
            **span["attrs"],
        }
        _stdout_log(record)
        if _LANGFUSE is not None:
            _LANGFUSE.span(
                name=name,
                metadata=span["attrs"],
                end_time=time.time(),
            )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _stdout_log(
            {
                "event": "trace_error",
                "name": name,
                "ms": round(elapsed_ms, 2),
                "error": repr(exc),
                **span["attrs"],
            }
        )
        raise
