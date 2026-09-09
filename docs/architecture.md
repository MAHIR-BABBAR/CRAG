# Architecture

CRAG turns a folder of documents into a retrieval service. It stops at packed
context with citations; generating an answer is the calling agent's job.

```
                 ingest                          query
  file ──► parse ──► chunk ──► embed ──► SQLite ──► BM25 ─┐
                     (parent/child)      (FTS5 +          ├─► RRF ─► rerank* ─► pack ─► context
                                          vectors) ─► dense ┘                            + citations
                                                                          * optional cross-encoder
```

## Ingest

**Parse.** A registry maps a file to the first parser that claims it. Each parser
emits `ContentBlock`s with a block type (heading, paragraph, code, table) and a
hierarchy path, so structure survives into retrieval. PDFs go through the
Unstructured API; everything else is parsed locally.

**Chunk.** Blocks become two kinds of chunk. A **parent** is a section-sized span
that reads well as context. A **child** is a smaller span sized for embedding.
Children carry a `parent_chunk_id`. Search runs over children, because small
spans embed sharply; what gets handed to the model is the parent, because a
model needs the surrounding sentences. See
[ADR 0002](adr/0002-parent-child-chunking.md).

**Embed.** Only children are embedded. A content-addressed cache keyed by
`(text_hash, model)` means re-indexing an unchanged file costs no API calls, and
a document whose content hash is unchanged is skipped entirely.

**Store.** One SQLite file holds documents, chunks, embeddings, and an FTS5
index over child text. Both retrieval channels read from the same file, so there
is nothing to keep in sync between a vector store and a keyword store. See
[ADR 0001](adr/0001-sqlite-as-the-index.md).

## Query

**BM25** runs against FTS5. The query is tokenized and OR-combined, which matters
for long natural-language queries where an AND of every term matches nothing.

**Dense** scores the query vector against every stored vector by cosine
similarity. This is exact, not approximate, in both the pure-Python and numpy
backends. See [ADR 0003](adr/0003-exact-dense-search.md).

**Fusion.** The two ranked lists are combined with Reciprocal Rank Fusion. RRF
uses ranks rather than scores, so it needs no calibration between a BM25 score
and a cosine similarity. See [ADR 0004](adr/0004-rrf-for-fusion.md).

**Rerank (optional).** A cross-encoder scores `(query, passage)` pairs directly
and reorders the candidates. It is accurate and slow, so it is off by default.

**Pack.** Selected chunks are assembled under a character budget, deduplicated by
parent, and emitted with numbered citations pointing back at `doc_id`,
`chunk_id`, source path, and page or slide where the parser recorded one.

**Quality label.** Every context response carries `empty`, `low`, or `high`.
Because each channel scores on its own scale, each has its own threshold: cosine
for dense, bm25 magnitude for lexical, and RRF score for hybrid, where the
default threshold is set at the point where both channels agree on a passage
rather than one channel guessing alone.

## Guardrails

**Embedding model invariance.** Vectors from different models are not
comparable, but cosine similarity between them still returns a confident-looking
ranking. Every collection records the `(provider, model, dimensions)` it was
built with, and a query embedded with a different model is refused rather than
answered badly.

**Index path restriction.** `/v1/index` reads from the server's filesystem. When
`api.allowed_index_roots` is set, paths are resolved first and then checked for
containment, so symlinks and `..` cannot escape.

## Layout

| Path | Responsibility |
|---|---|
| `core/` | Settings, types, exceptions, embedding providers |
| `ingestion/parsers/` | Format-specific parsing into content blocks |
| `ingestion/chunkers/` | Parent/child chunk construction |
| `ingestion/embedders/` | Batching, caching, content-hash dedup |
| `storage/` | SQLite schema, vector packing, BM25 and dense search |
| `retrieval/` | Channel search, RRF, rerank, packing, quality |
| `api/` | FastAPI socket, auth, dependency wiring |
| `eval/` | Datasets, metrics, runners, report rendering |
