"""Embed child chunks from a chunked document."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.config import get_settings
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.registry import resolve_embedding_provider
from custom_rag.core.types import Chunk, ChunkedDocument, EmbeddedChunk, EmbeddedDocument
from custom_rag.ingestion.embedders.cache import EmbeddingCache
from custom_rag.ingestion.embedders.config import EmbedConfig
from custom_rag.ingestion.embedders.dedup import DedupState, should_skip_embedding
from custom_rag.ingestion.parsers.base import content_hash


def embed_document(
    document: ChunkedDocument,
    *,
    provider: EmbeddingProvider | None = None,
    config: EmbedConfig | None = None,
    dedup_state: DedupState | None = None,
    cache: EmbeddingCache | None = None,
) -> EmbeddedDocument:
    active_config = config or EmbedConfig()
    settings = get_settings()
    batch_size = active_config.batch_size or settings.embedding.batch_size
    active_provider = provider or resolve_embedding_provider(
        settings,
        provider=active_config.provider,
        model=active_config.model,
    )
    active_cache = cache if cache is not None else EmbeddingCache()
    parents = document.parents()

    if should_skip_embedding(
        source_path=document.metadata.source_path,
        content_hash=document.metadata.content_hash,
        dedup_state=dedup_state,
    ):
        return EmbeddedDocument(
            metadata=document.metadata,
            children=[],
            parents=parents,
            skipped=True,
        )

    child_chunks = document.children()
    embedded_children = _embed_child_chunks(
        child_chunks,
        provider=active_provider,
        cache=active_cache,
        batch_size=batch_size,
    )

    if dedup_state is not None:
        dedup_state.set_content_hash(
            document.metadata.source_path,
            document.metadata.content_hash,
        )

    return EmbeddedDocument(
        metadata=document.metadata,
        children=embedded_children,
        parents=parents,
        skipped=False,
    )


def _embed_child_chunks(
    chunks: list[Chunk],
    *,
    provider: EmbeddingProvider,
    cache: EmbeddingCache,
    batch_size: int,
) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    model = provider.model
    pending: list[tuple[Chunk, str, str]] = []
    vectors_by_chunk_id: dict[str, list[float]] = {}

    for chunk in chunks:
        text_hash = content_hash(chunk.text.encode("utf-8"))
        cached = cache.get(text_hash, model)
        if cached is not None:
            vectors_by_chunk_id[chunk.chunk_id] = cached
        else:
            pending.append((chunk, text_hash, chunk.text))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [item[2] for item in batch]
        embeddings = provider.embed_texts(texts)
        for (chunk, text_hash, _text), embedding in zip(batch, embeddings, strict=True):
            cache.set(text_hash, model, embedding)
            vectors_by_chunk_id[chunk.chunk_id] = embedding

    embedded_at = datetime.now(tz=UTC)
    results: list[EmbeddedChunk] = []
    for chunk in chunks:
        embedding = vectors_by_chunk_id[chunk.chunk_id]
        text_hash = content_hash(chunk.text.encode("utf-8"))
        results.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                model=model,
                provider=provider.name,
                dimensions=provider.dimensions,
                embedded_at=embedded_at,
                text_hash=text_hash,
            )
        )
    return results
