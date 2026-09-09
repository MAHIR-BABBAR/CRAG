# CRAG eval report - `scifact`

- provider: `local`
- top_k: `10` documents
- docs: `5183`
- queries: `300`
- rerank: `False`

Metrics are per document: chunks are collapsed to their source document
before the list is cut to top_k. Brackets are 95% bootstrap intervals
over queries.

| mode | Hit@k | nDCG@k | Recall@k | MRR | p50 ms | p95 ms |
|------|------:|-------:|---------:|----:|-------:|-------:|
| vector | 0.830 [0.787, 0.870] | 0.671 [0.625, 0.713] | 0.815 | 0.631 | 704.4 | 939.4 |
| bm25 | 0.777 [0.730, 0.820] | 0.631 [0.586, 0.675] | 0.757 | 0.598 | 309.1 | 413.6 |
| hybrid | 0.857 [0.817, 0.893] | 0.690 [0.644, 0.729] | 0.840 | 0.649 | 998.0 | 1289.0 |

## Comparison

- hybrid - bm25 Hit@k: `+0.080`
- hybrid - vector Hit@k: `+0.027`
- hybrid - bm25 nDCG@k: `+0.059`
- hybrid - vector nDCG@k: `+0.018`

## Run

- commit: `1ef6197-dirty`
- crag: `0.2.0` | python: `3.14.0`
- platform: `Windows-11-10.0.26200-SP0`
- embedding model: `all-MiniLM-L6-v2` via `local`
- dense search: `numpy` | candidate_k: `20` | rrf_k: `60`
- chunks retrieved per requested document: `5`
- sentence-transformers: `5.4.1` | torch: `2.9.1`
- index time: `506.86s` | total: `1096.06s`
- started: `2026-09-09T14:24:05.512327+00:00`
- dataset: `beir/scifact/test` (full) fingerprint `60a24131166c3789`
