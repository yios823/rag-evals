# Observability for RAG systems

A RAG call has at least three stages (retrieve, rerank, generate) and silently failing or quietly degrading is the default if you do not instrument them. The minimum useful instrumentation:

- Per-stage spans with timing and the inputs that matter (query text, top-k chunk IDs, model version).
- The chunking config and embedding-model version recorded on every trace, so an answer from last week can be replayed against the exact retrieval state that produced it.
- Structured logs to stdout when no tracing backend is configured, so a developer running locally still sees what is happening.
- Errors recorded and re-raised, never swallowed.

Langfuse, Phoenix, Honeycomb, and OpenTelemetry-based stacks are all reasonable backends. The choice matters less than the discipline of emitting the spans in the first place.

A common mistake is to add observability after the system is already in production. By that point you are debugging blind for the first incident, which is exactly when you needed the traces. Wire them in from the first commit; the cost is a few lines per stage.
