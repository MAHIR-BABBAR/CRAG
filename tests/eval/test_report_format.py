"""The published report must state the conditions the numbers came from."""

from __future__ import annotations

import json
from pathlib import Path

from custom_rag.eval.report import build_comparison, report_to_markdown, write_report

REPORT: dict[str, object] = {
    "suite": "scifact",
    "provider": "local",
    "top_k": 10,
    "doc_count": 5183,
    "query_count": 300,
    "rerank": False,
    "dataset": {
        "dataset_id": "beir/scifact/test",
        "fingerprint": "60a24131166c3789",
        "subset": False,
    },
    "run": {
        "environment": {
            "crag_version": "0.2.0",
            "git_commit": "abc1234",
            "python": "3.12.3",
            "platform": "Linux-6.1",
            "packages": {"sentence-transformers": "3.0.1", "torch": "2.3.0"},
        },
        "retrieval": {
            "provider": "local",
            "embedding_model": "all-MiniLM-L6-v2",
            "candidate_k": 20,
            "rrf_k": 60,
            "chunk_multiplier": 5,
            "dense_search": "python",
        },
        "started_at": "2026-09-09T12:00:00+00:00",
        "index_seconds": 900.0,
        "total_seconds": 1200.0,
    },
    "modes": {
        "hybrid": {
            "hit_at_k": 0.8333,
            "recall_at_k": 0.8140,
            "mrr": 0.6400,
            "ndcg_at_k": 0.6750,
            "confidence_intervals": {
                "hit_at_k": {"low": 0.79, "high": 0.87},
                "ndcg_at_k": {"low": 0.63, "high": 0.72},
            },
            "latency_ms": {"p50_ms": 41.2, "p95_ms": 88.9},
        }
    },
}


def test_markdown_reports_intervals_and_latency() -> None:
    markdown = report_to_markdown(REPORT)
    assert "0.833 [0.790, 0.870]" in markdown
    assert "| 41.2 " in markdown
    assert "| 88.9 |" in markdown


def test_markdown_states_the_grading_unit() -> None:
    markdown = report_to_markdown(REPORT)
    assert "per document" in markdown
    assert "bootstrap" in markdown


def test_markdown_records_provenance() -> None:
    markdown = report_to_markdown(REPORT)
    assert "abc1234" in markdown
    assert "all-MiniLM-L6-v2" in markdown
    assert "beir/scifact/test" in markdown
    assert "(full)" in markdown


def test_subset_runs_are_labelled_as_subsets() -> None:
    report = json.loads(json.dumps(REPORT))
    report["dataset"]["subset"] = True
    assert "(subset)" in report_to_markdown(report)


def test_markdown_survives_a_report_without_a_run_block() -> None:
    minimal = {
        "suite": "fixtures",
        "modes": {"bm25": {"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0}},
    }
    markdown = report_to_markdown(minimal)
    assert "fixtures" in markdown
    assert "| bm25 " in markdown


def test_comparison_is_signed_relative_to_hybrid() -> None:
    comparison = build_comparison(
        {
            "hybrid": {"hit_at_k": 0.8, "ndcg_at_k": 0.7},
            "bm25": {"hit_at_k": 0.6, "ndcg_at_k": 0.5},
            "vector": {"hit_at_k": 0.9, "ndcg_at_k": 0.8},
        }
    )
    assert comparison["hybrid_minus_bm25_hit"] > 0
    assert comparison["hybrid_minus_vector_hit"] < 0


def test_write_report_emits_json_and_markdown(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    write_report(REPORT, target)
    assert json.loads(target.read_text(encoding="utf-8"))["suite"] == "scifact"
    assert "CRAG eval report" in target.with_suffix(".md").read_text(encoding="utf-8")
