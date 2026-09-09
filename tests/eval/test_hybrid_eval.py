"""Golden-set eval: hybrid should not lose to BM25 on lexical queries."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.config import CRAGSettings, EmbeddingSettings, StorageSettings
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.eval.runner import (
    default_fixture_paths,
    evaluate_mode,
    index_corpus,
    load_qrels,
)
from custom_rag.storage.sqlite_store import SQLiteIndexStore

EVAL_ROOT = Path(__file__).resolve().parent
FIXTURES = EVAL_ROOT.parent / "fixtures"
QRELS = EVAL_ROOT / "qrels.json"


def test_hybrid_matches_or_beats_bm25_on_lexical_qrels(tmp_path: Path) -> None:
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "eval.sqlite3"), collection="eval"),
    )
    provider = MockEmbeddingProvider()
    store = SQLiteIndexStore(tmp_path / "eval.sqlite3", default_collection="eval")
    try:
        index_corpus(
            default_fixture_paths(FIXTURES),
            store=store,
            provider=provider,
            collection="eval",
        )
        queries = [q for q in load_qrels(QRELS) if q.focus == "lexical"]
        assert len(queries) >= 5

        bm25 = evaluate_mode(
            queries,
            store=store,
            mode="bm25",
            provider=None,
            settings=settings,
            collection="eval",
            top_k=5,
        )
        hybrid = evaluate_mode(
            queries,
            store=store,
            mode="hybrid",
            provider=provider,
            settings=settings,
            collection="eval",
            top_k=5,
        )
        vector = evaluate_mode(
            queries,
            store=store,
            mode="vector",
            provider=provider,
            settings=settings,
            collection="eval",
            top_k=5,
        )

        # Lexical gold: BM25 and hybrid should both find the tokens.
        assert bm25.metrics.hit_at_k >= 0.8
        assert hybrid.metrics.hit_at_k >= bm25.metrics.hit_at_k - 1e-9
        assert hybrid.metrics.recall_at_k >= bm25.metrics.recall_at_k - 1e-9
        # Document mock-vector weakness: hybrid should beat pure vector on lexical set.
        assert hybrid.metrics.hit_at_k >= vector.metrics.hit_at_k - 1e-9
    finally:
        store.close()


def test_empty_query_marked_empty(tmp_path: Path) -> None:
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "eval.sqlite3"), collection="eval"),
    )
    provider = MockEmbeddingProvider()
    store = SQLiteIndexStore(tmp_path / "eval.sqlite3", default_collection="eval")
    try:
        index_corpus(
            default_fixture_paths(FIXTURES),
            store=store,
            provider=provider,
            collection="eval",
        )
        empty_qs = [q for q in load_qrels(QRELS) if q.expect_empty]
        assert empty_qs
        report = evaluate_mode(
            empty_qs,
            store=store,
            mode="bm25",
            provider=None,
            settings=settings,
            collection="eval",
            top_k=5,
        )
        assert report.metrics.hit_at_k == 1.0
        assert report.rows[0].relevant_in_top_k == 0
    finally:
        store.close()
