# ADR 0002: Search small chunks, return large ones

Status: accepted

## Context

Chunk size is a direct tradeoff. Small chunks embed sharply, because one
paragraph has one topic and its vector points somewhere specific. Large chunks
read better, because an answer usually needs the sentences around the match.
Picking one size means losing one of those properties.

## Decision

Build two chunk layers from the same parse. **Children** are embedding-sized and
are the only thing indexed and searched. **Parents** are section-sized and are
what gets returned and packed into context. Every child stores its
`parent_chunk_id`.

## Why

The tradeoff is only forced if retrieval and presentation have to use the same
unit. They do not. Search over children keeps the vectors discriminative; return
parents so the model sees a coherent passage rather than a sentence that
begins with "It does not."

Deduplicating by parent during packing also stops three children of the same
section from consuming the whole context budget with overlapping text.

## Cost

Storage roughly doubles, since parent text is kept alongside child text. Parent
lookup adds a query per hit, which is a primary-key read.

There is a subtler cost in evaluation: because several children can belong to
one document, counting chunks as results would inflate recall. The eval harness
collapses chunks to documents before scoring; see
[docs/evaluation.md](../evaluation.md).

## Alternatives

**One medium chunk size.** Simpler, and worse at both ends.

**Sliding windows with overlap.** Recovers some context but duplicates text
across vectors, which spends embedding cost on near-duplicates and makes the
same passage compete with itself in the ranking.
