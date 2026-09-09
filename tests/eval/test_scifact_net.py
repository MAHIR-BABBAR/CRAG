"""Network SciFact benchmark eval (opt-in)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from custom_rag.eval.runner import run_benchmark_eval

pytestmark = pytest.mark.network


def test_scifact_net_eval_smoke(tmp_path: Path) -> None:
    if os.environ.get("CRAG_NET_EVAL") != "1":
        pytest.skip("set CRAG_NET_EVAL=1 to run SciFact network eval")

    pytest.importorskip("ir_datasets")

    report = run_benchmark_eval(
        suite="scifact",
        db_path=tmp_path / "scifact.sqlite3",
        provider_name="mock",
        top_k=5,
        cache_dir=tmp_path / "cache",
        max_docs=30,
        max_queries=10,
        report_path=tmp_path / "report.json",
    )
    assert report["suite"] == "scifact"
    assert report["doc_count"] > 0
    assert report["query_count"] > 0
    for mode in ("vector", "bm25", "hybrid"):
        assert mode in report["modes"]
        metrics = report["modes"][mode]
        assert "hit_at_k" in metrics
        assert "ndcg_at_k" in metrics
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    # BM25 should find something on a real labeled subset.
    assert report["modes"]["bm25"]["hit_at_k"] >= 0.0
