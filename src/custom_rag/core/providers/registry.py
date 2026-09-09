"""Embedding provider registry."""

from __future__ import annotations

from custom_rag.core.config import CRAGSettings, EmbeddingSettings
from custom_rag.core.exceptions import ConfigError, EmbeddingError
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.local import LocalEmbeddingProvider
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.providers.openai import OpenAIEmbeddingProvider


def resolve_embedding_provider(
    settings: CRAGSettings | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    active_settings = settings or CRAGSettings()
    embedding = active_settings.embedding
    provider_name = provider or embedding.provider
    model_name = model or embedding.model

    if provider_name == "mock":
        return MockEmbeddingProvider(model=model_name, dimensions=embedding.dimensions or 8)

    if provider_name == "openai":
        try:
            return OpenAIEmbeddingProvider(
                api_key=active_settings.openai_api_key or "",
                model=model_name,
                dimensions=embedding.dimensions,
            )
        except EmbeddingError as exc:
            raise ConfigError(str(exc)) from exc

    if provider_name == "local":
        return LocalEmbeddingProvider(model=model_name, dimensions=embedding.dimensions)

    msg = f"unsupported embedding provider: {provider_name!r}"
    raise ConfigError(msg)


def resolve_embedding_provider_from_settings(
    embedding: EmbeddingSettings,
    *,
    openai_api_key: str | None = None,
) -> EmbeddingProvider:
    settings = CRAGSettings(embedding=embedding)
    settings.openai_api_key = openai_api_key
    return resolve_embedding_provider(settings)
