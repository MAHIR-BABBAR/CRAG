"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from custom_rag.core.config import (
    HARD_MAX_CANDIDATE_K,
    HARD_MAX_TOP_K,
    ApiSettings,
    ContextSettings,
    CRAGSettings,
    EmbeddingSettings,
    RetrievalSettings,
    StorageSettings,
    load_settings,
)


def test_load_settings_reads_yaml_embedding_section(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "embedding:\n  provider: mock\n  model: mock-embed-v1\n  batch_size: 32\n"
        "storage:\n  backend: sqlite\n  path: tmp/crag.sqlite3\n  collection: docs\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    assert settings.embedding.provider == "mock"
    assert settings.embedding.model == "mock-embed-v1"
    assert settings.embedding.batch_size == 32
    assert settings.storage.path == "tmp/crag.sqlite3"
    assert settings.storage.collection == "docs"


def test_crag_settings_env_overrides_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_MODEL", "mock-test")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("CRAG_STORAGE_PATH", "custom/index.sqlite3")
    monkeypatch.setenv("CRAG_COLLECTION", "research")

    settings = CRAGSettings()
    assert settings.embedding.provider == "mock"
    assert settings.embedding.model == "mock-test"
    assert settings.embedding.batch_size == 16
    assert settings.storage.path == "custom/index.sqlite3"
    assert settings.storage.collection == "research"


def test_embedding_settings_defaults() -> None:
    settings = EmbeddingSettings()
    assert settings.provider == "local"
    assert settings.model == "all-MiniLM-L6-v2"
    assert settings.batch_size == 64


def test_storage_settings_defaults() -> None:
    settings = StorageSettings()
    assert settings.backend == "sqlite"
    assert settings.path == "workspace_index/crag.sqlite3"
    assert settings.collection == "default"
    assert settings.dense_search == "python"


def test_retrieval_settings_defaults() -> None:
    settings = RetrievalSettings()
    assert settings.mode == "hybrid"
    assert settings.top_k == 5
    assert settings.candidate_k == 20
    assert settings.rrf_k == 60
    assert settings.max_top_k == 100
    assert settings.max_candidate_k == 200


def test_crag_settings_env_overrides_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("CRAG_TOP_K", "7")
    monkeypatch.setenv("CRAG_CANDIDATE_K", "30")

    settings = CRAGSettings()
    assert settings.retrieval.mode == "hybrid"
    assert settings.retrieval.top_k == 7
    assert settings.retrieval.candidate_k == 30


def test_api_settings_defaults() -> None:
    settings = ApiSettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.api_key is None
    assert settings.allowed_index_roots == []


def test_allowed_index_roots_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_ALLOWED_INDEX_ROOTS", os.pathsep.join(["/srv/docs", "/data"]))
    settings = CRAGSettings()
    assert settings.api.allowed_index_roots == ["/srv/docs", "/data"]


def test_dense_search_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_DENSE_SEARCH", "numpy")
    assert CRAGSettings().storage.dense_search == "numpy"


def test_retrieval_depth_caps_cannot_exceed_hard_limits() -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(max_top_k=HARD_MAX_TOP_K + 1)
    with pytest.raises(ValidationError):
        RetrievalSettings(max_candidate_k=HARD_MAX_CANDIDATE_K + 1)


def test_crag_settings_env_overrides_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_API_HOST", "0.0.0.0")
    monkeypatch.setenv("CRAG_API_PORT", "9000")

    settings = CRAGSettings()
    assert settings.api.host == "0.0.0.0"
    assert settings.api.port == 9000


def test_context_settings_defaults() -> None:
    settings = ContextSettings()
    assert settings.max_chars == 4000
    assert settings.include_parents is True
    assert settings.min_score_high == 0.25


def test_crag_settings_env_overrides_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_CONTEXT_MAX_CHARS", "2048")
    settings = CRAGSettings()
    assert settings.context.max_chars == 2048
