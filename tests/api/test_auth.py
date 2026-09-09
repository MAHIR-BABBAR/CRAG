"""API key auth tests."""

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


def _settings(tmp_path: Path, *, api_key: str | None) -> CRAGSettings:
    return CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3")),
        api=ApiSettings(host="127.0.0.1", port=8000, api_key=api_key),
    )


def test_health_open_when_api_key_set(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, api_key="secret"))) as client:
        assert client.get("/v1/health").status_code == 200


def test_index_requires_api_key(tmp_path: Path) -> None:
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(_settings(tmp_path, api_key="secret"))) as client:
        denied = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert denied.status_code == 401

        ok = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200

        ok2 = client.post(
            "/v1/retrieve",
            json={"query": "hello", "provider": "mock", "mode": "bm25"},
            headers={"X-API-Key": "secret"},
        )
        assert ok2.status_code == 200
