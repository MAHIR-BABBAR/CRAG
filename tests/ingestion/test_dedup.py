"""Tests for document-level dedup during embedding."""

from __future__ import annotations

from datetime import UTC, datetime

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.types import BlockType, Chunk, ChunkedDocument, ChunkRole, DocumentMetadata
from custom_rag.ingestion.embedders.dedup import InMemoryDedupState
from custom_rag.ingestion.embedders.embedder import embed_document


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc-1",
        source_path="/tmp/sample.txt",
        source_uri="file:///tmp/sample.txt",
        doc_type="text",
        content_hash="hash-v1",
        file_size=128,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _chunked_document() -> ChunkedDocument:
    return ChunkedDocument(
        metadata=_metadata(),
        chunks=[
            Chunk(
                chunk_id="parent_doc_root",
                role=ChunkRole.PARENT,
                text="Document",
                source_block_id="doc_root",
                block_type=BlockType.DOCUMENT,
            ),
            Chunk(
                chunk_id="child_p_0",
                role=ChunkRole.CHILD,
                text="First paragraph.",
                source_block_id="p_0",
                parent_chunk_id="parent_doc_root",
                block_type=BlockType.PARAGRAPH,
            ),
        ],
    )


def test_dedup_skips_reembedding_when_content_hash_unchanged() -> None:
    dedup_state = InMemoryDedupState()
    dedup_state.set_content_hash("/tmp/sample.txt", "hash-v1")
    provider = MockEmbeddingProvider()

    embedded = embed_document(
        _chunked_document(),
        provider=provider,
        dedup_state=dedup_state,
    )

    assert embedded.skipped is True
    assert embedded.children == []
    assert len(embedded.parents) == 1


def test_dedup_embeds_when_content_hash_changes() -> None:
    dedup_state = InMemoryDedupState()
    dedup_state.set_content_hash("/tmp/sample.txt", "old-hash")
    provider = MockEmbeddingProvider()

    embedded = embed_document(
        _chunked_document(),
        provider=provider,
        dedup_state=dedup_state,
    )

    assert embedded.skipped is False
    assert len(embedded.children) == 1
    assert dedup_state.get_content_hash("/tmp/sample.txt") == "hash-v1"
