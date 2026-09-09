"""Shared FastAPI dependencies for the open socket."""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from custom_rag.core.config import CRAGSettings, EmbeddingProviderName
from custom_rag.core.exceptions import ConfigError
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.registry import resolve_embedding_provider
from custom_rag.storage.sqlite_store import SQLiteIndexStore


class ProviderCache:
    """Keeps one provider instance per name for the process lifetime.

    The local provider holds a sentence-transformers model; building a new one
    per request would reload several hundred megabytes of weights each time.
    """

    def __init__(self, settings: CRAGSettings) -> None:
        self._settings = settings
        self._providers: dict[str, EmbeddingProvider] = {}

    def get(self, provider_name: EmbeddingProviderName | None) -> EmbeddingProvider:
        key = provider_name or self._settings.embedding.provider
        cached = self._providers.get(key)
        if cached is not None:
            return cached
        provider = resolve_embedding_provider(self._settings, provider=key)
        self._providers[key] = provider
        return provider


def get_settings(request: Request) -> CRAGSettings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_store(request: Request) -> SQLiteIndexStore:
    return request.app.state.store  # type: ignore[no-any-return]


def get_provider_cache(request: Request) -> ProviderCache:
    return request.app.state.providers  # type: ignore[no-any-return]


def resolve_index_path(path_value: str, settings: CRAGSettings) -> Path:
    """Resolve a client-supplied index path inside the configured roots.

    Resolution happens before the containment check so that symlinks and ``..``
    segments cannot be used to walk out of an allowed root.
    """
    resolved = Path(path_value).expanduser().resolve()
    roots = settings.api.allowed_index_roots
    if not roots:
        return resolved
    for root in roots:
        allowed_root = Path(root).expanduser().resolve()
        if resolved == allowed_root or allowed_root in resolved.parents:
            return resolved
    raise ConfigError(f"path is outside the allowed index roots: {', '.join(sorted(roots))}")
