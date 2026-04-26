---
title: rag-evals demo
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.40.0
app_file: app/gradio_app.py
pinned: false
---

# rag-evals demo

Interactive demo for [yios823/rag-evals](https://github.com/yios823/rag-evals).
Hybrid retrieval (dense + BM25) → cross-encoder rerank → Claude with structured output and citation validation.

## Set the secret

In Space settings → Variables and secrets, add `ANTHROPIC_API_KEY`. The Space will not start without it.

## Build the index

The Space's startup runs `make ingest` automatically (see `app/setup.sh`) so the sample corpus is indexed on first boot.
