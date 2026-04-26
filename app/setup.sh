#!/usr/bin/env bash
# Run on Hugging Face Space startup. Idempotent.
set -euo pipefail
if [ ! -f data/index/chunks.jsonl ]; then
  python -m rag_evals.ingest --corpus data/corpus --out data/index
fi
