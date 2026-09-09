# Contributing to CRAG

## Setup

```bash
pip install -e ".[dev,ingestion,embeddings,storage,numpy]"
```

## Checks

What CI runs, and what you should run before opening a PR:

```bash
ruff check src tests scripts examples
ruff format --check src tests scripts examples
mypy src
pytest -q --ignore=tests/eval/test_scifact_net.py --cov=custom_rag --cov-fail-under=80
```

Network SciFact is optional and intentionally excluded from CI:

```bash
pip install -e ".[eval]"
# PowerShell: $env:CRAG_NET_EVAL="1"
CRAG_NET_EVAL=1 pytest tests/eval/test_scifact_net.py -m network
```

## Style

- Prefer small, focused PRs
- Do not add in-library LLM answer generation — agents own generation
- Dense search is exact (`python` or `numpy`), not an ANN index; keep naming honest

## Docs

Update `CHANGELOG.md` for user-visible changes. Architecture decisions go in
`docs/adr/`. Evaluation methodology lives in `docs/evaluation.md`; do not claim
BEIR-leaderboard comparability for CRAG's chunked pipeline.
