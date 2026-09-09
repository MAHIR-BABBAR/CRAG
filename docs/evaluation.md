# How CRAG is evaluated

This describes the method. The numbers are in [benchmarks.md](benchmarks.md).

The point of writing this down is that retrieval metrics are easy to inflate by
accident, and a number without its method is not worth much.

## What is measured

Retrieval only: given a query, does the right document come back near the top.
Nothing here measures answer quality, because the library does not generate
answers ([ADR 0005](adr/0005-retrieval-only.md)).

## The dataset

[BEIR SciFact](https://github.com/beir-cellar/beir), test split, pulled with
`ir_datasets` as `beir/scifact/test`: 5,183 abstracts and 300 judged claims.
Relevance comes from the published qrels, matched on the dataset's own document
IDs, which are carried through indexing as `external_id`. No relevance judgement
in the benchmark is written by this project.

Queries whose relevant documents are not in the corpus are dropped, and qrels
with relevance 0 are not treated as positive.

## Grading unit: documents, not chunks

This is the correction that matters most.

CRAG indexes chunks, so one document can occupy several positions in a ranked
list. Scoring those positions independently inflates everything: a document split
into ten chunks can fill the entire top 10 and count as ten retrieved relevant
results.

So the harness **collapses chunks to their source document before cutting to
top_k**. A document ranks where its best chunk ranked, appears once, and is
relevant if any of its retrieved chunks satisfies the qrels. To leave room for
that collapsing, retrieval pulls `top_k × chunk_multiplier` chunks (default 5)
and the collapsed list is then truncated to `top_k` documents.

`Hit@10` therefore means "a relevant document is among the top 10 distinct
documents returned", which is what the same metric means in the IR literature.

## Metrics

- **Hit@k** — fraction of queries with at least one relevant document in top k.
- **Recall@k** — relevant documents found, over relevant documents that exist,
  capped at 1.0 per query.
- **MRR** — mean of 1/rank of the first relevant document.
- **nDCG@k** — discounted gain against the ideal ordering, gold-aware, so a query
  with one relevant document has an ideal DCG of one document.

Every metric is computed per query and then averaged, which is what makes the
confidence intervals possible.

## Confidence intervals

Reported alongside each metric is a 95% percentile bootstrap interval over
queries: resample the 300 queries with replacement 1,000 times, recompute the
mean, take the 2.5th and 97.5th percentiles. The seed is fixed, so the interval
is reproducible.

With 300 queries, differences of one or two points are inside the noise. The
interval is there so nobody reads a 0.008 gap as a result.

## Rerank ablation

Both arms of the reranking comparison retrieve **the same candidate pool**, and
differ only in whether the cross-encoder reorders it. Letting the reranked arm
see a deeper pool would credit the reranker for candidates the baseline was never
shown, and that is the most common way an ablation flatters a reranker.

## Cache integrity

Prepared corpora are fingerprinted by dataset ID, `max_docs`, and `max_queries`,
and subsets are cached in their own directory. A 100-document smoke run cannot
leave a truncated corpus behind for the next full run to report against.

## Provenance

Each report records the commit, Python version, platform, embedding model,
sentence-transformers and torch versions, retrieval settings, dataset
fingerprint, and index and total wall time. Reports are written as JSON with a
rendered Markdown companion.

## What these numbers are not

They are **not comparable to the BEIR leaderboard**. Leaderboard entries index
whole abstracts as single units; CRAG chunks them and collapses back, which is a
different retrieval system over the same data. The comparison that is valid here
is between CRAG's own modes on identical data, which is the comparison the
reports make.

SciFact is also one domain — scientific claim verification, short abstracts,
formal language. Results on code, chat logs, or long PDFs would differ, and this
project has not measured those.

## Reproducing

```bash
pip install -e ".[dev,ingestion,embeddings,storage,eval,numpy]"

# Full run (~30 min on CPU the first time, mostly embedding 5,183 abstracts).
# Reuse --db-path on later arms so indexing is not paid again.
crag eval --suite scifact --provider local --top-k 10 \
  --db-path workspace_index/bench.sqlite3 \
  --report reports/scifact_baseline.json

# Same index, cross-encoder rerank on top
crag eval --suite scifact --provider local --top-k 10 --rerank \
  --db-path workspace_index/bench.sqlite3 \
  --report reports/scifact_rerank.json

# Fast subset, for checking the harness rather than the retriever
crag eval --suite scifact --provider mock --max-docs 300 --max-queries 30 \
  --report reports/scifact_smoke.json
```

Set `CRAG_DENSE_SEARCH=numpy` for the vectorized exact backend. Published
numbers and interpretation live in [benchmarks.md](benchmarks.md).

The offline fixture suite (`--suite fixtures`) runs in seconds against
`tests/fixtures` and is what CI exercises; it checks that the harness works, not
that retrieval is good.
