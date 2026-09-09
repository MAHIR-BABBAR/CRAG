# ADR 0001: SQLite as the whole index

Status: accepted

## Context

CRAG needs to store documents, chunks, embeddings, and a keyword index, and it
needs lexical and dense search over the same corpus. The obvious approach is a
vector database next to a search engine.

## Decision

Keep everything in one SQLite file. Vectors are float32 blobs; lexical search
uses the built-in FTS5 extension.

## Why

Two stores means two things that can disagree. A document deleted from one and
left in the other produces results that cite text that no longer exists, and
that class of bug is invisible until someone reads the output carefully. With a
single file, an upsert deletes and rewrites a document's rows and FTS entries in
one transaction: either both channels see the new version or neither does.

It also removes the setup step. `pip install` and a path is the whole
dependency list, which is what makes the demo and the test suite runnable
anywhere without a container or a running service.

## Cost

Dense search is a full scan, so this does not hold at millions of chunks; see
[ADR 0003](0003-exact-dense-search.md) for where that line sits. SQLite also
serializes writes, which is fine for indexing batches and would not be fine for
high-frequency concurrent writes.

## Alternatives

**pgvector.** Better ceiling and real concurrency, but it needs a server, which
costs the zero-setup property that makes this project easy to evaluate.

**FAISS plus a keyword index.** Fast, but it is the two-store consistency problem
plus a second persistence format to manage.
