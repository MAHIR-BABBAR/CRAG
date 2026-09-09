"""Document-level collapsing, confidence intervals, and cache fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_rag.core.config import CRAGSettings, EmbeddingSettings, StorageSettings
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.eval.datasets import SCIFACT_DATASET_ID, corpus_fingerprint
from custom_rag.eval.metrics import bootstrap_ci, evaluate_ranking
from custom_rag.eval.runner import EvalQuery, RelevantRule, evaluate_mode, run_eval_suite
from custom_rag.ingestion import index_file
from custom_rag.storage.sqlite_store import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
QRELS = Path(__file__).resolve().parent / "qrels.json"


def test_many_chunks_of_one_document_count_once(tmp_path: Path) -> None:
    """A long document must not fill the result list with itself."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    long_doc = "\n\n".join(
        f"Section {i}. This paragraph mentions ZenithProtocol in passing." for i in range(12)
    )
    (corpus / "long.md").write_text(f"# Long\n\n{long_doc}", encoding="utf-8")
    (corpus / "other.md").write_text("Unrelated filler about gardening.", encoding="utf-8")

    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "eval.sqlite3"), collection="eval"),
    )
    provider = MockEmbeddingProvider()
    with SQLiteIndexStore(tmp_path / "eval.sqlite3", default_collection="eval") as store:
        for path in sorted(corpus.iterdir()):
            index_file(path, provider=provider, store=store, collection="eval")

        report = evaluate_mode(
            [
                EvalQuery(
                    id="q1",
                    query="ZenithProtocol",
                    relevant=[RelevantRule(source_contains="long.md")],
                )
            ],
            store=store,
            mode="bm25",
            provider=None,
            settings=settings,
            collection="eval",
            top_k=5,
        )

    row = report.rows[0]
    assert row.hit_ids == sorted(set(row.hit_ids), key=row.hit_ids.index)
    # One gold document retrieved once, so recall is exactly 1.0, never above it.
    assert row.relevant_in_top_k == 1
    assert report.metrics.recall_at_k == pytest.approx(1.0)
    assert report.metrics.ndcg_at_k <= 1.0


def test_document_relevance_survives_chunk_ordering(tmp_path: Path) -> None:
    """A document is relevant if any retrieved chunk matches, not just the top one."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text(
        "# Intro\n\nOpening paragraph with the query term Nimbus.\n\n"
        "## Details\n\nThe answer phrase Quasar lives further down.\n",
        encoding="utf-8",
    )
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "eval.sqlite3"), collection="eval"),
    )
    with SQLiteIndexStore(tmp_path / "eval.sqlite3", default_collection="eval") as store:
        index_file(
            corpus / "doc.md",
            provider=MockEmbeddingProvider(),
            store=store,
            collection="eval",
        )
        report = evaluate_mode(
            [
                EvalQuery(
                    id="q1",
                    query="Nimbus",
                    relevant=[RelevantRule(text_contains="Quasar")],
                )
            ],
            store=store,
            mode="bm25",
            provider=None,
            settings=settings,
            collection="eval",
            top_k=5,
        )
    assert report.metrics.hit_at_k == 1.0


def test_reported_metrics_carry_confidence_intervals(tmp_path: Path) -> None:
    report = run_eval_suite(
        qrels_path=QRELS,
        fixtures_dir=FIXTURES,
        db_path=tmp_path / "eval.sqlite3",
        provider_name="mock",
        top_k=5,
    )
    hybrid = report["modes"]["hybrid"]
    intervals = hybrid["confidence_intervals"]
    assert set(intervals) == {"hit_at_k", "recall_at_k", "mrr", "ndcg_at_k"}
    for metric, bounds in intervals.items():
        assert bounds["low"] <= hybrid[metric] <= bounds["high"]
    assert hybrid["latency_ms"]["p95_ms"] >= hybrid["latency_ms"]["p50_ms"]


def test_report_records_the_environment_it_ran_in(tmp_path: Path) -> None:
    report = run_eval_suite(
        qrels_path=QRELS,
        fixtures_dir=FIXTURES,
        db_path=tmp_path / "eval.sqlite3",
        provider_name="mock",
        top_k=5,
    )
    run = report["run"]
    assert run["environment"]["crag_version"]
    assert run["environment"]["python"]
    assert run["retrieval"]["embedding_model"] == "mock-embed-v1"
    assert run["retrieval"]["chunk_multiplier"] >= 1
    assert run["grading"]["unit"] == "document"
    # The report must be serializable, since that is how it gets published.
    json.dumps(report)


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [1.0] * 70 + [0.0] * 30
    interval = bootstrap_ci(values, iterations=500)
    assert interval.low < 0.7 < interval.high
    assert interval.low >= 0.0
    assert interval.high <= 1.0


def test_bootstrap_interval_is_deterministic() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert bootstrap_ci(values, iterations=200) == bootstrap_ci(values, iterations=200)


def test_unanimous_scores_give_a_degenerate_interval() -> None:
    interval = bootstrap_ci([1.0] * 20, iterations=200)
    assert interval.low == pytest.approx(1.0)
    assert interval.high == pytest.approx(1.0)


def test_expected_empty_query_scores_perfectly_when_empty() -> None:
    assert evaluate_ranking([], gold_count=0, k=5).hit == 1.0
    assert evaluate_ranking([1.0], gold_count=0, k=5).hit == 0.0


def test_ndcg_cannot_exceed_one_with_extra_hits() -> None:
    assert evaluate_ranking([1.0, 1.0, 1.0], gold_count=1, k=3).ndcg == pytest.approx(1.0)


def test_cache_fingerprint_separates_subsets_from_full_runs() -> None:
    full = corpus_fingerprint(SCIFACT_DATASET_ID, None, None)
    smoke = corpus_fingerprint(SCIFACT_DATASET_ID, 100, 20)
    assert full != smoke
    assert full == corpus_fingerprint(SCIFACT_DATASET_ID, None, None)
