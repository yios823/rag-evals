# Rules for Claude Code working on this repo

## What this project is

A small reference RAG implementation that exercises hybrid retrieval, cross-encoder rerank, structured generator output, and a CI-gated eval suite. Tiny on purpose — the value is the *shape*, not scale. Read `README.md` before changing anything.

## Hard invariants (don't break these)

- **Generator output is always Pydantic-validated.** Free-text downstream of `src/rag_evals/generate.py:Answer` is forbidden. If a check needs new fields, extend the model — never bypass it.
- **Citations are membership-checked against retrieved context.** If the model cites a chunk ID that wasn't in the retrieved set, `_parse()` raises `ValueError`. Don't relax this — it catches a class of hallucination that scoring misses.
- **Pinned model versions live in `src/rag_evals/config.py:CONFIG`.** Changing any field there changes retrieval/generation behavior, which can move eval scores. Always re-run `make eval` after touching.
- **CI gates on `eval_thresholds.yaml`.** A PR that drops `recall@5`, `mrr@10`, `ctx_precision@5`, `faithfulness`, or `refusal_correct` below the floor cannot merge. `hallucination_rate` is a max, not a min — flip the sign in your head.
- **Observability never silently fails.** `src/rag_evals/observability.py` re-raises errors after recording; never swallow.

## Layout

- `src/rag_evals/` — module code; `eval/` subpackage holds metrics + CLI runner
- `data/corpus/` — sample markdown corpus (5 files); `data/golden.jsonl` — labeled queries
- `tests/` — pytest unit tests (chunking + metrics); CI runs these on every PR
- `.github/workflows/evals.yml` — CI: unit tests → ingest → eval (mock generator) → render dashboard → publish to GitHub Pages on main
- `dashboards/render.py` — turns `data/results.json` into static HTML for Pages

## Code style

- Line length 100 (set in `pyproject.toml`, ruff). The IDE's flake8 default of 79 is noise — ignore unless I ask for tighter.
- Type hints on every public function; `from __future__ import annotations` at the top of every module.
- `tenacity` for retries on external calls (Anthropic). Never an unbounded `while True`.

## When adding features

1. Add the eval first if it changes retrieval/generation quality. If you can't measure it, it doesn't ship.
2. Extend `golden.jsonl` if a new query class isn't represented.
3. Run `make eval` locally before pushing. Fix threshold regressions before commit, not after.
4. Update `README.md` "What's inside" map if file layout changed.

## When debugging eval regressions

1. Diff the per-query records in `data/results.json` between runs (look for which qids dropped).
2. Check if `CONFIG` changed — embedding/reranker/generator model version is the most common culprit.
3. Walk the trace logs (stdout JSON when no Langfuse) — `retrieve.hybrid` span shows top-1 chunk per stage.
