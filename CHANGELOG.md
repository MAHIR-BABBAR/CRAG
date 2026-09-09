# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-09

### Added
- MIT `LICENSE`, GitHub Actions CI (lint, types, coverage, wheel build, Docker smoke)
- Recursive directory indexing (`crag index ./docs`, `POST /v1/index` with dirs)
- Optional API key auth (`CRAG_API_KEY`) for index/retrieve/context
- `api.allowed_index_roots` so the HTTP API can only index whitelisted paths
- Vectorized dense backend (`storage.dense_search: numpy`) alongside the pure-Python one
- Guard that rejects queries whose embedding model differs from the indexed one
- Provider cache so the API loads a sentence-transformers model once, not per request
- Run provenance, 95% bootstrap confidence intervals, and latency percentiles in eval reports
- Docker demo stack with a baked-in model, non-root user, and health check
- `demo_corpus/`, `examples/agent_client.py`, CONTRIBUTING, `docs/` with architecture and ADRs

### Changed
- Default retrieval mode is hybrid, which is the best-scoring mode on SciFact
- Evaluation collapses chunk hits to documents before scoring, so metrics are document-level
- Rerank ablation draws both arms from the same candidate pool for a fair comparison
- Eval dataset caches are keyed by a corpus fingerprint, so subsets cannot clobber full runs
- Renamed `storage.ann` to `storage.dense_search`; the values are now `python` and `numpy`,
  which describe what the backends actually are: exact scans, not approximate indexes

### Fixed
- BM25 tokenization no longer requires every query term to match, which was zeroing out
  recall on long claims
- BM25 failures raise `StorageError` instead of silently returning no results
- nDCG can no longer exceed 1.0 when several chunks of one document are retrieved
- CLI commands close their SQLite store instead of leaking the connection

### Measured (SciFact test, local MiniLM, top_k=10, no rerank)
- hybrid Hit@10 **0.857**, Recall@10 **0.840**, MRR **0.649**, nDCG@10 **0.690**
- dense 0.830 Hit@10, BM25 0.777; full results in `docs/benchmarks.md`

## [0.1.0] - 2026-08-30

### Added
- Parse → parent/child chunk → embed → SQLite index
- Hybrid BM25 ∥ dense → RRF; modes `vector` / `bm25` / `hybrid`
- FastAPI socket: `/v1/health`, `/v1/index`, `/v1/retrieve`, `/v1/context`
- Context packer with citations + retrieval quality labels
- Eval harness (fixtures + SciFact via `ir_datasets`)
- Optional cross-encoder rerank (opt-in)

[0.2.0]: https://github.com/MAHIR-BABBAR/CRAG/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MAHIR-BABBAR/CRAG/releases/tag/v0.1.0
