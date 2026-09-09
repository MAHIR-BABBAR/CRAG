"""API directory index contract."""

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


def test_index_directory_returns_batch(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Alpha uniqueTokenA", encoding="utf-8")
    (docs / "b.md").write_text("# Beta uniqueTokenB", encoding="utf-8")
    settings = CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3")),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/index",
            json={"path": str(docs), "provider": "mock", "recursive": True},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["indexed"] == 2
        assert payload["failed"] == []
        assert payload["file_count"] == 2
