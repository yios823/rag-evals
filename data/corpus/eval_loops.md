# RAG evaluation loops

A RAG eval suite has two layers. Retrieval metrics measure whether the right context reached the generator. Answer metrics measure what the generator did with that context.

Retrieval metrics worth tracking:

- `recall @ k`: of the documents you labeled as relevant for a query, how many appear in the top k retrieved? This is the most important upstream signal; if recall is low, no amount of prompt engineering on the generator will fix the system.
- `MRR @ k`: 1 / rank of the first relevant document. Sensitive to the tail of cases where the relevant document is found but is buried.
- `context precision @ k`: of the top k retrieved documents, how many are relevant? High precision matters more when the generator's context window is tight.

Answer metrics worth tracking:

- Faithfulness: are the claims in the answer supported by the retrieved context? Implementations range from cheap lexical overlap to LLM-judge scoring. Pick one and report which.
- Hallucination rate: how often does the generator make up facts or cite sources that were not in its context. The citation-membership check at the generator boundary catches the citation case directly.
- Refusal correctness: when the corpus genuinely cannot answer the question, does the model refuse? Adversarial unanswerable items in the golden set are how you measure this.

The eval suite should run as a CLI, write its results to a file, and gate CI on minimum thresholds. Notebook-based evals drift the moment two engineers run them on different days.
