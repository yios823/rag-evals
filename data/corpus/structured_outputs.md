# Structured generator outputs

When an LLM's response feeds into downstream code, free text is a bug magnet. The model can decide to add a preface, switch from JSON to YAML mid-output, or invent fields. Each of those produces a runtime exception in the consuming code and a long tail of brittle regex parsers.

The right pattern is to define the response shape as a Pydantic model (or equivalent JSON Schema), pass that schema to the model in the prompt, and validate the response against it before any downstream code sees it. Anthropic, OpenAI, and most modern providers support a "JSON mode" or schema-constrained generation that increases the rate of well-formed output, but validation at the boundary is still required because providers cannot guarantee schema adherence.

A useful refinement: in addition to validating the JSON shape, validate that any IDs the model returns actually came from the context the model was given. If a RAG generator cites `chunk-77` and `chunk-77` was not in the retrieved set, the model is hallucinating its citation. Failing the call at that boundary catches a class of hallucination that pure faithfulness scoring misses.
