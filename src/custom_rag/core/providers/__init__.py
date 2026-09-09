"""Multi-provider embedding registry."""

from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.registry import (
    resolve_embedding_provider,
    resolve_embedding_provider_from_settings,
)

__all__ = [
    "EmbeddingProvider",
    "resolve_embedding_provider",
    "resolve_embedding_provider_from_settings",
]
