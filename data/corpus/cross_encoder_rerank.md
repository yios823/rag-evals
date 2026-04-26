# Cross-encoder reranking

Bi-encoder retrieval encodes the query and each document independently, then compares vectors. This is fast and indexable but coarse: the model never sees the query and the document together, so it cannot reason about their interaction.

A cross-encoder reranker takes pairs of (query, candidate document) and produces a single relevance score per pair. Because the model attends jointly over both texts, it captures fine-grained relevance signals that bi-encoders miss. The cost is that it cannot be precomputed, so it only runs over a small candidate set returned by the first-stage retriever.

A typical setup retrieves the top 20 to 100 candidates with a hybrid bi-encoder + BM25 stage, then reranks with a cross-encoder to keep the top 5 or 10. The `cross-encoder/ms-marco-MiniLM-L-6-v2` model is a good default: it is small, fast on CPU, and well-trained on a passage-ranking benchmark.

The win shows up most clearly in `context precision @ k` and `recall @ k` for small k. If your eval set shows the right document is usually in the top 20 but rarely in the top 3, a reranker is the right next step.
