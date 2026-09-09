"""Tests for embedding providers."""

from __future__ import annotations

import pytest

from custom_rag.core.config import CRAGSettings, EmbeddingSettings
from custom_rag.core.exceptions import ConfigError
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.providers.registry import resolve_embedding_provider


def test_mock_provider_returns_normalized_vectors() -> None:
    provider = MockEmbeddingProvider()
    vectors = provider.embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == provider.dimensions
    assert vectors[0] != vectors[1]


def test_mock_provider_is_deterministic() -> None:
    provider = MockEmbeddingProvider()
    first = provider.embed_texts(["repeat me"])
    second = provider.embed_texts(["repeat me"])
    assert first == second


def test_registry_resolves_mock_provider() -> None:
    settings = CRAGSettings(embedding=EmbeddingSettings(provider="mock", model="mock-v1"))
    provider = resolve_embedding_provider(settings)
    assert provider.name == "mock"
    assert provider.model == "mock-v1"


def test_registry_rejects_unknown_provider() -> None:
    settings = CRAGSettings(embedding=EmbeddingSettings(provider="mock"))
    with pytest.raises(ConfigError):
        resolve_embedding_provider(settings, provider="unknown")
