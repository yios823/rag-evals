# Hybrid retrieval

Dense retrievers built on bi-encoders are strong on paraphrase and topical similarity but weak on rare tokens, exact phrase matches, and product or version identifiers. BM25 has the opposite shape: it shines on precise lexical matches and degrades on paraphrase.

Hybrid retrieval runs both a dense search and a BM25 search, then merges the rankings. Reciprocal-rank fusion is the simplest stable merge: each document's fused score is the sum across rankings of `1 / (k + rank)`, where `k` is a smoothing constant (60 is a common default). Documents that appear high in either ranking float to the top; documents only seen by one retriever still get partial credit.

Weighted-sum schemes (e.g. `alpha * dense_score + (1 - alpha) * bm25_score`) can outperform RRF after careful tuning, but they require the two scores to be on a comparable scale and the alpha to be re-tuned whenever the index or models change. RRF avoids that maintenance cost, which is why it is a sensible default until eval numbers say otherwise.
