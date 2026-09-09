"""API contract tests for POST /v1/context."""

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

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _settings(tmp_path: Path) -> CRAGSettings:
    return CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3"), collection="default"),
        retrieval=RetrievalSettings(mode="vector", top_k=3),
        context=ContextSettings(max_chars=4000, include_parents=True, min_score_high=0.0),
        api=ApiSettings(host="127.0.0.1", port=8000),
    )


def test_context_empty_index_returns_empty_quality(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/v1/context",
            json={"query": "anything", "provider": "mock", "mode": "vector"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_quality"] == "empty"
    assert payload["context"] == ""
    assert payload["citations"] == []
    assert payload["quality_stats"]["hit_count"] == 0


def test_context_after_index_returns_packed_citations(tmp_path: Path) -> None:
    fixture = FIXTURES / "sample.txt"
    with TestClient(create_app(_settings(tmp_path))) as client:
        indexed = client.post(
            "/v1/index",
            json={"path": str(fixture), "provider": "mock"},
        )
        assert indexed.status_code == 200

        response = client.post(
            "/v1/context",
            json={
                "query": "First paragraph",
                "provider": "mock",
                "mode": "vector",
                "top_k": 3,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_quality"] in {"low", "high"}
    assert payload["context"]
    assert "[1]" in payload["context"]
    assert payload["citations"]
    assert payload["citations"][0]["source_path"]
    assert payload["quality_stats"]["chars_used"] <= 4000
    assert payload["hits"]
