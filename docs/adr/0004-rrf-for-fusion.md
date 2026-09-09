# ADR 0004: Reciprocal Rank Fusion for combining channels

Status: accepted

## Context

BM25 and dense search have to be merged into one ranking. Their scores are not
comparable: SQLite's `bm25()` is negative and unbounded, cosine similarity is
roughly `[-1, 1]`, and neither distribution is stable across queries or corpora.

## Decision

Fuse with Reciprocal Rank Fusion: each document scores `sum(1 / (k + rank))`
over the channels that returned it, with `k = 60`.

## Why

RRF reads ranks, not scores, so there is nothing to calibrate. A weighted sum of
scores would need per-corpus normalization, and that normalization would quietly
rot as the corpus changed.

It also degrades sensibly. A document ranked highly by both channels beats one
ranked highly by a single channel, which is the behaviour you want when one
channel is confidently wrong. `k = 60` is the value from the original Cormack et
al. work and flattens the difference between ranks 1 and 2 enough that a
narrow win in one channel does not dominate the fused list.

That same property gives the retrieval-quality label something to measure. At
`k = 60`, one channel ranking a passage first scores 1/61 ≈ 0.016, and both
channels doing so scores about 0.033. The default `high` threshold of 0.025 sits
between them, so `high` means the channels agreed.

## Cost

RRF discards score magnitude, so a dense hit at 0.95 and one at 0.55 are
interchangeable if they share a rank. When magnitudes genuinely matter, the fix
is the cross-encoder reranker, which scores pairs directly.

## Alternatives

**Weighted score sum.** Can beat RRF once tuned, and the tuning is per corpus.

**Learned fusion.** Needs training data this project does not have, and would
make the ranking hard to explain.
