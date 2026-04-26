# rag-evals

A small, runnable reference for the parts of a RAG system that usually break in front of users: hybrid retrieval, cross-encoder reranking, structured generator output, and an eval suite that runs in CI.

The corpus and golden questions are tiny on purpose. Drop your own markdown into `data/corpus/`, write your own questions into `data/golden.jsonl`, and the same pipeline runs end to end.

## Quick start

```bash
git clone https://github.com/yios823/rag-evals
cd rag-evals
make setup                 # creates .venv, installs deps
cp .env.example .env       # then add ANTHROPIC_API_KEY
make ingest                # chunks + embeds the sample corpus
make eval                  # runs the eval suite, writes data/results.json
make dashboard             # builds an HTML report into dashboards/build/
make demo                  # launches the Gradio interactive demo
```

## What's inside

```
src/rag_evals/
  ingest.py          token-aware chunking; builds FAISS dense index + BM25
  retrieve.py        hybrid retrieval with reciprocal-rank fusion
  rerank.py          cross-encoder reranker (ms-marco-MiniLM)
  generate.py        Claude call + Pydantic-validated answer + citation check
  observability.py   Langfuse if a key is set, structured stdout JSON otherwise
  config.py          pinned model versions and chunking config
  eval/
    golden.py        Pydantic schema for the labeled eval set
    metrics.py       recall@k, MRR, context precision, faithfulness
    run.py           CLI runner: golden.jsonl in, results.json out

data/
  corpus/            five short markdown docs used as the sample corpus
  golden.jsonl       sample labeled questions

.github/workflows/evals.yml   runs the eval suite on every PR; fails on regression
eval_thresholds.yaml          minimum scores the CI gate enforces
dashboards/render.py          turns results.json into an HTML report (Plotly)
app/gradio_app.py             interactive demo, deployable to Hugging Face Spaces
```

## Design notes

Why these specific pieces:

- **Hybrid over dense-only**: BM25 catches keyword and rare-term queries that dense embeddings smear. Reciprocal-rank fusion is cheap and stable; weighted-sum tuning is a tax I avoid until evals say it pays off.
- **Cross-encoder rerank**: bi-encoder retrieval is fast but coarse. A cross-encoder over the top-N candidates buys real precision on the top-K that the generator sees, and the cost is bounded.
- **Pydantic on the generator output**: the LLM doesn't get to invent its own response shape. Free text downstream is where most "weird production behavior" comes from.
- **Citation membership check**: if the generator cites a chunk ID that wasn't in the retrieved context, the call fails. This catches a class of hallucination that string-match faithfulness scoring misses.
- **Eval suite is a CLI, not a notebook**: same JSON powers the dashboard, the CI gate, and any future regression tracking. One source of truth.
- **Observability is wired in, not bolted on later**: every retrieve / rerank / generate call emits a span. Langfuse if you've configured it, structured JSON to stdout if you haven't. The trace records the embedding model, chunking config, and top-1 chunk per stage so you can replay any answer.

## What's not here, on purpose

- No fine-tuning. This is an inference-time stack.
- No multi-tenant infrastructure (auth, rate-limiting, per-user state). That's a layer above this.
- No production-scale storage. The FAISS index is in-memory. The retriever interface is small enough to swap for pgvector / Qdrant / OpenSearch when you need to.
- One generator backend by default (Anthropic). The shape of `generate.py` makes adding OpenAI / Bedrock a small change; the eval harness is generator-agnostic.

## Running the eval suite

```bash
make ingest
make eval
```

Output:

```
$ make eval
… running 12 golden queries against the index …
recall@5      : 0.83
mrr@10        : 0.71
ctx_precision : 0.78
faithfulness  : 0.91
hallucination : 0.00
✓ wrote data/results.json
```

The CI workflow runs the same command on every pull request. If any score drops below the threshold in `eval_thresholds.yaml`, the build fails and the PR can't merge.

## Bringing your own corpus

```bash
# replace the sample corpus
rm data/corpus/*.md
cp /path/to/your/docs/*.md data/corpus/

# write some labeled questions
$EDITOR data/golden.jsonl

# rebuild and re-evaluate
make ingest && make eval
```

The golden file is JSON Lines, one question per line. Schema is in `src/rag_evals/eval/golden.py`.

## License

MIT. See [LICENSE](LICENSE).
