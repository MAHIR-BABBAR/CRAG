# ADR 0003: Exact dense search, and saying so

Status: accepted

## Context

Dense retrieval here compares a query vector against every stored vector. An
earlier version of this project exposed that setting as `storage.ann` with
values `brute` and `numpy`, which implied an approximate nearest-neighbour index.

There is no ANN index. Both paths compute exact cosine similarity over the full
set. The numpy path is the same arithmetic as one matrix product instead of a
Python loop.

## Decision

Rename the setting to `storage.dense_search` with values `python` and `numpy`,
and document both as exact. Add a test that asserts the two backends produce
identical rankings and scores, so the claim is checked rather than asserted.

## Why

The old name would have survived exactly one interview question. Worse, it would
have set the wrong expectation about behaviour: an approximate index trades
recall for speed, and someone who believed this was ANN would have gone looking
for a recall knob that does not exist, or assumed results were approximate when
they are exact.

Naming it for what it does also makes the actual tradeoff visible: this is
O(n) per query in the number of chunks.

## Where the ceiling is

Exact cosine over a few hundred thousand chunks stays interactive. Past roughly a
million, the full scan dominates latency and a real ANN index — HNSW via
`sqlite-vec`, or an external store — becomes the right answer. The interface
that would need to change is `IndexStore.search_vector`, which is one method.

## Alternatives

**Ship an ANN index now.** Adds a dependency and an index-build step to buy
headroom the project does not currently need, and gives up exact results in
exchange.

**Keep the `ann` name.** Faster to leave alone, and wrong.
