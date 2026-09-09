"""Ingestion entry points: parse, chunk, embed, and persist a single file."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.types import (
    ChunkedDocument,
    DocumentMetadata,
    EmbeddedDocument,
    ParsedDocument,
)
from custom_rag.ingestion.chunkers.chunker import chunk_document
from custom_rag.ingestion.chunkers.config import ChunkConfig
from custom_rag.ingestion.embedders.config import EmbedConfig
from custom_rag.ingestion.embedders.dedup import DedupState
from custom_rag.ingestion.embedders.embedder import embed_document
from custom_rag.ingestion.metadata.file_metadata import extract_file_metadata
from custom_rag.ingestion.parsers.composite import build_default_registry
from custom_rag.ingestion.parsers.registry import ParserRegistry
from custom_rag.storage.base import IndexStore

_DEFAULT_REGISTRY = build_default_registry()


def parse_file(path: Path | str, *, registry: ParserRegistry | None = None) -> ParsedDocument:
    resolved = Path(path).expanduser().resolve()
    active_registry = registry or _DEFAULT_REGISTRY

    file_metadata = extract_file_metadata(resolved)
    parser = active_registry.get_parser(resolved, file_metadata.mime_type)
    document = parser.parse(resolved, file_metadata)
    return _merge_file_metadata(document, file_metadata)


def chunk_file(
    path: Path | str,
    *,
    registry: ParserRegistry | None = None,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    return chunk_document(parse_file(path, registry=registry), config=config)


def embed_file(
    path: Path | str,
    *,
    registry: ParserRegistry | None = None,
    chunk_config: ChunkConfig | None = None,
    embed_config: EmbedConfig | None = None,
    provider: EmbeddingProvider | None = None,
    dedup_state: DedupState | None = None,
) -> EmbeddedDocument:
    chunked = chunk_file(path, registry=registry, config=chunk_config)
    return embed_document(
        chunked,
        provider=provider,
        config=embed_config,
        dedup_state=dedup_state,
    )


def index_file(
    path: Path | str,
    *,
    registry: ParserRegistry | None = None,
    chunk_config: ChunkConfig | None = None,
    embed_config: EmbedConfig | None = None,
    provider: EmbeddingProvider | None = None,
    store: IndexStore | None = None,
    collection: str | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> EmbeddedDocument:
    """Embed a file and persist it to the local index."""
    from custom_rag.core.config import get_settings
    from custom_rag.storage.sqlite_store import SQLiteIndexStore

    settings = get_settings()
    active_store = store or SQLiteIndexStore(
        settings.storage.path,
        default_collection=settings.storage.collection,
    )
    active_collection = collection or settings.storage.collection
    embedded = embed_file(
        path,
        registry=registry,
        chunk_config=chunk_config,
        embed_config=embed_config,
        provider=provider,
        dedup_state=active_store,
    )
    if extra_metadata and not embedded.skipped:
        merged_extra = {**embedded.metadata.extra, **extra_metadata}
        embedded = embedded.model_copy(
            update={"metadata": embedded.metadata.model_copy(update={"extra": merged_extra})}
        )
    if not embedded.skipped:
        active_store.upsert(embedded, collection=active_collection)
    return embedded


def _merge_file_metadata(
    document: ParsedDocument, file_metadata: DocumentMetadata
) -> ParsedDocument:
    merged_extra = {**file_metadata.extra, **document.metadata.extra}
    encoding = document.metadata.encoding or file_metadata.encoding

    merged = file_metadata.model_copy(
        update={
            "doc_type": document.metadata.doc_type or file_metadata.doc_type,
            "language": document.metadata.language or file_metadata.language,
            "encoding": encoding,
            "extra": merged_extra,
        }
    )
    return document.model_copy(update={"metadata": merged})
