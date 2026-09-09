# CRAG Demo — Getting Started

CRAG is a **local retrieval service for agents**. It indexes documents, runs hybrid
search (BM25 + dense + RRF), and returns packed context with citations.

Agents own prompting and answer generation — CRAG does not call an LLM for answers.

## Why hybrid?

Lexical search catches exact terms; dense search catches paraphrases. Reciprocal Rank
Fusion combines both without needing a trained fusion model.
