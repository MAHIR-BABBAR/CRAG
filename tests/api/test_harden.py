"""Input-validation and failure-mode tests for the open socket."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from custom_rag.api.app import create_app
from custom_rag.api.deps import ProviderCache
from custom_rag.core.config import (
    ApiSettings,
    CRAGSettings,
    EmbeddingSettings,
    RetrievalSettings,
    StorageSettings,
)
from custom_rag.core.providers.openai import OpenAIEmbeddingProvider
from custom_rag.storage.sqlite_store import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _settings(
    tmp_path: Path,
    *,
    retrieval: RetrievalSettings | None = None,
    embedding: EmbeddingSettings | None = None,
    collection: str = "default",
) -> CRAGSettings:
    return CRAGSettings(
        embedding=embedding or EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(
            path=str(tmp_path / "crag.sqlite3"),
            collection=collection,
        ),
        retrieval=retrieval or RetrievalSettings(),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )


def test_retrieve_uses_injected_settings_defaults(tmp_path: Path) -> None:
    """Omitted mode/collection must come from app.state.settings, not global cache."""
    settings = _settings(
        tmp_path,
        retrieval=RetrievalSettings(mode="bm25", top_k=2),
        collection="injected",
    )
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(settings)) as client:
        index_response = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock", "collection": "injected"},
        )
        assert index_response.status_code == 200

        retrieve_response = client.post(
            "/v1/retrieve",
            json={"query": "paragraph"},
        )
        assert retrieve_response.status_code == 200
        payload = retrieve_response.json()
        assert payload["mode"] == "bm25"
        assert len(payload["hits"]) <= 2


def test_bm25_retrieve_without_working_embedder(tmp_path: Path) -> None:
    """BM25 must not require resolving a broken embedding provider."""
    settings = _settings(
        tmp_path,
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        retrieval=RetrievalSettings(mode="bm25"),
    )
    # No OPENAI_API_KEY — vector/hybrid would fail; bm25 must still work.
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(settings)) as client:
        index_response = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert index_response.status_code == 200

        retrieve_response = client.post(
            "/v1/retrieve",
            json={"query": "paragraph", "mode": "bm25"},
        )
        assert retrieve_response.status_code == 200
        assert retrieve_response.json()["mode"] == "bm25"
        assert retrieve_response.json()["hits"]


def test_top_k_cap_rejected(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/v1/retrieve",
            json={"query": "hello", "provider": "mock", "top_k": 101},
        )
    assert response.status_code == 422


def test_skipped_index_returns_stored_counts(tmp_path: Path) -> None:
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(_settings(tmp_path))) as client:
        first = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert first.status_code == 200
        assert first.json()["skipped"] is False
        children = first.json()["children"]
        parents = first.json()["parents"]
        doc_id = first.json()["doc_id"]

        second = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert second.status_code == 200
        skipped = second.json()
        assert skipped["skipped"] is True
        assert skipped["doc_id"] == doc_id
        assert skipped["children"] == children
        assert skipped["parents"] == parents


def test_health_degraded_when_store_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        store: SQLiteIndexStore = app.state.store
        store.close()
        response = client.get("/v1/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_provider_cache_uses_injected_openai_settings() -> None:
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-large"),
        openai_api_key="sk-test-injected",
    )
    provider = ProviderCache(settings).get("openai")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.model == "text-embedding-3-large"
    assert provider._api_key == "sk-test-injected"


def test_provider_cache_reuses_one_instance_per_provider() -> None:
    """The local provider owns a loaded model; rebuilding it per call is costly."""
    cache = ProviderCache(
        CRAGSettings(embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"))
    )
    assert cache.get("mock") is cache.get("mock")
    assert cache.get(None) is cache.get("mock")


def test_concurrent_index_and_retrieve_smoke(tmp_path: Path) -> None:
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(_settings(tmp_path))) as client:
        # Seed once so retrieve has content.
        seed = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert seed.status_code == 200

        def index_once() -> int:
            return client.post(
                "/v1/index",
                json={"path": str(fixture), "provider": "mock"},
            ).status_code

        def retrieve_once() -> int:
            return client.post(
                "/v1/retrieve",
                json={"query": "paragraph", "provider": "mock", "mode": "vector"},
            ).status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(index_once) for _ in range(4)] + [
                pool.submit(retrieve_once) for _ in range(4)
            ]
            statuses = [future.result() for future in futures]
        assert all(status == 200 for status in statuses)
