"""Index and retrieve endpoint contract tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from custom_rag.api.app import create_app
from custom_rag.core.config import (
    ApiSettings,
    CRAGSettings,
    EmbeddingSettings,
    StorageSettings,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _settings(tmp_path: Path) -> CRAGSettings:
    return CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3"), collection="default"),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )


def test_index_missing_path_returns_404(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/v1/index",
            json={"path": str(tmp_path / "missing.txt"), "provider": "mock"},
        )
    assert response.status_code == 404


def test_index_then_retrieve_returns_hits_with_parents(tmp_path: Path) -> None:
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(_settings(tmp_path))) as client:
        index_response = client.post(
            "/v1/index",
            json={
                "path": str(fixture),
                "provider": "mock",
                "collection": "default",
            },
        )
        assert index_response.status_code == 200
        indexed = index_response.json()
        assert indexed["skipped"] is False
        assert indexed["children"] > 0
        assert indexed["parents"] > 0
        assert indexed["doc_id"]

        retrieve_response = client.post(
            "/v1/retrieve",
            json={
                "query": "First paragraph",
                "provider": "mock",
                "mode": "vector",
                "top_k": 3,
                "collection": "default",
            },
        )
        assert retrieve_response.status_code == 200
        payload = retrieve_response.json()
        assert payload["query"] == "First paragraph"
        assert payload["mode"] == "vector"
        assert payload["hits"]
        top = payload["hits"][0]
        assert top["child"]["text"]
        assert top["parent"] is not None
        assert top["parent"]["text"]
