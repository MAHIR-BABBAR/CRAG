"""Health endpoint contract tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from custom_rag.api.app import create_app
from custom_rag.core.config import ApiSettings, CRAGSettings, StorageSettings


def test_health_returns_ok(tmp_path: Path) -> None:
    settings = CRAGSettings(
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3"), collection="default"),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["collection"] == "default"
    assert payload["storage_path"] == str(tmp_path / "crag.sqlite3")
