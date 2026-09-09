# CRAG eval report - `scifact`

- provider: `local`
- top_k: `10` documents
- docs: `5183`
- queries: `300`
- rerank: `True`

Metrics are per document: chunks are collapsed to their source document
before the list is cut to top_k. Brackets are 95% bootstrap intervals
over queries.

| mode | Hit@k | nDCG@k | Recall@k | MRR | p50 ms | p95 ms |
|------|------:|-------:|---------:|----:|-------:|-------:|
| vector | 0.837 [0.793, 0.877] | 0.693 [0.647, 0.735] | 0.826 | 0.659 | 2047.8 | 3758.2 |
| bm25 | 0.803 [0.757, 0.847] | 0.679 [0.634, 0.723] | 0.784 | 0.657 | 1742.3 | 1938.6 |
| hybrid | 0.853 [0.817, 0.890] | 0.700 [0.655, 0.739] | 0.842 | 0.663 | 2142.1 | 2392.0 |

## Comparison

- hybrid - bm25 Hit@k: `+0.050`
- hybrid - vector Hit@k: `+0.017`
- hybrid - bm25 nDCG@k: `+0.020`
- hybrid - vector nDCG@k: `+0.006`

## Run

- commit: `1ef6197-dirty`
- crag: `0.2.0` | python: `3.14.0`
- platform: `Windows-11-10.0.26200-SP0`
- embedding model: `all-MiniLM-L6-v2` via `local`
- dense search: `numpy` | candidate_k: `20` | rrf_k: `60`
- chunks retrieved per requested document: `5`
- sentence-transformers: `5.4.1` | torch: `2.9.1`
- index time: `13.62s` | total: `1951.77s`
- started: `2026-09-09T14:34:10.184008+00:00`
- dataset: `beir/scifact/test` (full) fingerprint `60a24131166c3789`
