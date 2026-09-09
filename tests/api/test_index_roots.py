"""/v1/index must stay inside the configured roots."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from custom_rag.api.app import create_app
from custom_rag.api.deps import resolve_index_path
from custom_rag.core.config import (
    ApiSettings,
    CRAGSettings,
    EmbeddingSettings,
    StorageSettings,
)
from custom_rag.core.exceptions import ConfigError


def _settings(tmp_path: Path, roots: list[str]) -> CRAGSettings:
    return CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(tmp_path / "crag.sqlite3"), collection="default"),
        api=ApiSettings(host="127.0.0.1", port=8000, allowed_index_roots=roots),
    )


def test_path_inside_allowed_root_is_indexed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha content", encoding="utf-8")

    with TestClient(create_app(_settings(tmp_path, [str(corpus)]))) as client:
        response = client.post(
            "/v1/index",
            json={"path": str(corpus / "a.md"), "provider": "mock"},
        )
    assert response.status_code == 200


def test_path_outside_allowed_root_is_forbidden(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    outside = tmp_path / "secrets.md"
    outside.write_text("do not index me", encoding="utf-8")

    with TestClient(create_app(_settings(tmp_path, [str(corpus)]))) as client:
        response = client.post(
            "/v1/index",
            json={"path": str(outside), "provider": "mock"},
        )
    assert response.status_code == 403
    assert "allowed index roots" in response.json()["detail"]


def test_traversal_out_of_allowed_root_is_forbidden(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (tmp_path / "secrets.md").write_text("do not index me", encoding="utf-8")

    with TestClient(create_app(_settings(tmp_path, [str(corpus)]))) as client:
        response = client.post(
            "/v1/index",
            json={"path": str(corpus / ".." / "secrets.md"), "provider": "mock"},
        )
    assert response.status_code == 403


def test_no_roots_configured_allows_any_path(tmp_path: Path) -> None:
    target = tmp_path / "anywhere.md"
    target.write_text("content", encoding="utf-8")
    settings = _settings(tmp_path, [])
    assert resolve_index_path(str(target), settings) == target.resolve()


def test_resolve_rejects_sibling_prefix_collision(tmp_path: Path) -> None:
    """`/data/corpus-private` must not pass a `/data/corpus` allow-list."""
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    sibling = tmp_path / "corpus-private"
    sibling.mkdir()
    settings = _settings(tmp_path, [str(allowed)])

    with pytest.raises(ConfigError):
        resolve_index_path(str(sibling / "leak.md"), settings)
