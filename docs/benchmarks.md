# Benchmark results

Method is in [evaluation.md](evaluation.md). Read that first if a number here
looks surprising; most surprises are about how the metric is defined.

All runs: BEIR SciFact test split, 5,183 abstracts, 300 claims, `all-MiniLM-L6-v2`,
`top_k=10` documents, `candidate_k=20` per channel, RRF `k=60`, chunk multiplier 5.
Brackets are 95% bootstrap confidence intervals over the 300 queries.
Raw reports with full provenance are in [`reports/`](../reports/).

## Retrieval modes

From [`reports/scifact_baseline.md`](../reports/scifact_baseline.md):

| mode | Hit@10 | nDCG@10 | Recall@10 | MRR | p50 ms | p95 ms |
|------|-------:|--------:|----------:|----:|-------:|-------:|
| bm25 | 0.777 [0.730, 0.820] | 0.631 [0.586, 0.675] | 0.757 | 0.598 | 309 | 414 |
| vector | 0.830 [0.787, 0.870] | 0.671 [0.625, 0.713] | 0.815 | 0.631 | 704 | 939 |
| **hybrid** | **0.857 [0.817, 0.893]** | **0.690 [0.644, 0.729]** | **0.840** | **0.649** | 998 | 1289 |

Hybrid beats BM25 by 8.0 points of Hit@10 and dense by 2.7. The margin over BM25
is comfortably outside the confidence intervals. The margin over dense alone is
not: the intervals overlap heavily, so the honest claim is that hybrid is at
least as good as dense on this dataset and clearly better than lexical alone,
and that it wins on every metric rather than by a decisive gap on any one.

That is the expected shape for SciFact. Claims are paraphrases of abstract
sentences, which favours dense retrieval, but they retain scientific terminology
that BM25 matches exactly. Fusion collects both without a trained combiner.

## Cross-encoder reranking

`ms-marco-MiniLM-L-6-v2` reordering the same candidate pool the baseline saw,
from [`reports/scifact_rerank.md`](../reports/scifact_rerank.md):

| mode | Hit@10 | ΔHit | nDCG@10 | ΔnDCG | MRR | ΔMRR | p50 ms |
|------|-------:|-----:|--------:|------:|----:|-----:|-------:|
| bm25 | 0.803 | +0.026 | 0.679 | +0.048 | 0.657 | +0.059 | 1742 |
| vector | 0.837 | +0.007 | 0.693 | +0.022 | 0.659 | +0.028 | 2048 |
| hybrid | 0.853 | −0.004 | 0.700 | +0.010 | 0.663 | +0.014 | 2142 |

Reranking reorders; it does not retrieve. Hit@10 barely moves, and on hybrid it
moves down by an amount that is pure noise, because the candidate pool is fixed
and the reranker can only shuffle what fusion already found. What it does change
is ordering quality: MRR and nDCG rise everywhere, most for BM25, which had the
worst ordering to begin with.

The cost is roughly 2× latency. So reranking earns its place on a weak single
channel, and on hybrid it buys a small ordering improvement for double the
budget. It is off by default, which is why.

The general pattern holds: the more work fusion has already done, the less a
reranker adds.

## Dense backend parity

The two dense backends are both exact cosine over every stored vector, so they
should return identical rankings and differ only in speed. They do, which is the
point of running both — [`reports/scifact_dense_python.md`](../reports/scifact_dense_python.md)
against the baseline above:

| backend | hybrid Hit@10 | hybrid nDCG@10 | vector p50 ms | hybrid p50 ms |
|---------|--------------:|---------------:|--------------:|--------------:|
| `python` | 0.857 | 0.690 | 1391 | 1922 |
| `numpy` | 0.857 | 0.690 | 704 | 998 |

Every metric matches to three decimals across all three modes; numpy is about
2× faster. Note what the remaining latency is: a query still reads every stored
vector out of SQLite and decodes it, and that deserialization, not the dot
product, is what a full scan costs. Vectorizing the arithmetic can only remove
the arithmetic. Escaping the rest needs an ANN index
([ADR 0003](adr/0003-exact-dense-search.md)).

Latency was measured on a Windows laptop CPU with no GPU, single process. Treat
the absolute numbers as a shape, not a spec, and the ratios as the real result.

## Incremental indexing

Building the index from scratch took 507s. Re-running the same corpus against
the existing index took **13.6s**, because unchanged documents are skipped on
their content hash before parsing and unchanged chunks hit the embedding cache.
Both figures are in the provenance blocks of the two reports.

## What is not measured

No answer quality, since the library does not generate answers. No domain other
than scientific abstracts. No corpus large enough to make exact dense search
hurt, which is precisely where the current design would fail first.
