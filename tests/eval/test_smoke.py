"""End-to-end smoke: index fixtures → retrieve → context pack."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from custom_rag.api.app import create_app
from custom_rag.core.config import (
    ApiSettings,
    ContextSettings,
    CRAGSettings,
    EmbeddingSettings,
    RetrievalSettings,
    StorageSettings,
)
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion import index_file
from custom_rag.retrieval import assemble_context, retrieve_with_parents, score_retrieval
from custom_rag.storage.sqlite_store import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_smoke_index_retrieve_context_library(tmp_path: Path) -> None:
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "smoke.sqlite3"), collection="smoke"),
        context=ContextSettings(max_chars=2000, include_parents=True, min_score_high=0.0),
    )
    store = SQLiteIndexStore(tmp_path / "smoke.sqlite3", default_collection="smoke")
    provider = MockEmbeddingProvider()
    try:
        for name in ("sample.md", "sample.txt"):
            embedded = index_file(
                FIXTURES / name,
                provider=provider,
                store=store,
                collection="smoke",
            )
            assert embedded.skipped is False

        result = retrieve_with_parents(
            "Install the package with pip",
            store=store,
            provider=provider,
            mode="hybrid",
            top_k=3,
            collection="smoke",
            settings=settings,
        )
        assert result.hits
        packed = assemble_context(
            result.hits,
            max_chars=settings.context.max_chars,
            include_parents=True,
            source_path_for=store.get_source_path,
        )
        quality = score_retrieval(
            result.hits,
            packed,
            mode=result.mode,
            min_score_high=settings.context.min_score_high,
        )
        assert packed.text
        assert packed.citations
        assert quality.label in {"low", "high"}
        assert "[1]" in packed.text
    finally:
        store.close()


def test_smoke_api_index_and_context(tmp_path: Path) -> None:
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "api-smoke.sqlite3"), collection="smoke"),
        retrieval=RetrievalSettings(mode="hybrid", top_k=3),
        context=ContextSettings(max_chars=2000, include_parents=True, min_score_high=0.0),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )
    with TestClient(create_app(settings)) as client:
        for name in ("sample.md", "sample.txt"):
            resp = client.post(
                "/v1/index",
                json={"path": str(FIXTURES / name), "provider": "mock", "collection": "smoke"},
            )
            assert resp.status_code == 200
            assert resp.json()["skipped"] is False

        context_resp = client.post(
            "/v1/context",
            json={
                "query": "First paragraph",
                "provider": "mock",
                "mode": "hybrid",
                "collection": "smoke",
            },
        )
        assert context_resp.status_code == 200
        payload = context_resp.json()
        assert payload["context"]
        assert payload["citations"]
        assert payload["retrieval_quality"] != "empty"
