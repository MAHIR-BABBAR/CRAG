"""Querying an index with the wrong embedding model must fail loudly."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from custom_rag.core.exceptions import ConfigError
from custom_rag.core.types import (
    BlockType,
    Chunk,
    ChunkRole,
    DocumentMetadata,
    EmbeddedChunk,
    EmbeddedDocument,
)
from custom_rag.retrieval import retrieve_with_parents
from custom_rag.storage.sqlite_store import SQLiteIndexStore


class _Provider:
    name = "mock"

    def __init__(self, model: str, dimensions: int = 8) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dimensions - 1) for _ in texts]


def _seed(store: SQLiteIndexStore, model: str) -> None:
    child = Chunk(
        chunk_id="child_0",
        role=ChunkRole.CHILD,
        text="hybrid retrieval combines lexical and dense channels",
        source_block_id="p_0",
        parent_chunk_id=None,
        block_type=BlockType.PARAGRAPH,
    )
    store.upsert(
        EmbeddedDocument(
            metadata=DocumentMetadata(
                doc_id="doc-1",
                source_path="a.txt",
                source_uri="file://a.txt",
                doc_type="text",
                content_hash="hash-1",
                file_size=10,
                modified_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            parents=[],
            children=[
                EmbeddedChunk(
                    chunk=child,
                    embedding=[1.0] + [0.0] * 7,
                    model=model,
                    provider="mock",
                    dimensions=8,
                    embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                    text_hash="th-1",
                )
            ],
        )
    )


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_mismatched_query_model_is_rejected(tmp_path: Path, mode: str) -> None:
    with SQLiteIndexStore(tmp_path / "crag.sqlite3") as store:
        _seed(store, "all-MiniLM-L6-v2")
        with pytest.raises(ConfigError) as excinfo:
            retrieve_with_parents(
                "hybrid retrieval",
                store=store,
                provider=_Provider("text-embedding-3-small"),  # type: ignore[arg-type]
                mode=mode,  # type: ignore[arg-type]
                top_k=3,
            )
    message = str(excinfo.value)
    assert "all-MiniLM-L6-v2" in message
    assert "text-embedding-3-small" in message


def test_matching_query_model_is_allowed(tmp_path: Path) -> None:
    with SQLiteIndexStore(tmp_path / "crag.sqlite3") as store:
        _seed(store, "all-MiniLM-L6-v2")
        result = retrieve_with_parents(
            "hybrid retrieval",
            store=store,
            provider=_Provider("all-MiniLM-L6-v2"),  # type: ignore[arg-type]
            mode="vector",
            top_k=3,
        )
    assert result.hits


def test_bm25_needs_no_embedding_model(tmp_path: Path) -> None:
    """Lexical search compares text, so a model mismatch is irrelevant to it."""
    with SQLiteIndexStore(tmp_path / "crag.sqlite3") as store:
        _seed(store, "all-MiniLM-L6-v2")
        result = retrieve_with_parents(
            "hybrid retrieval",
            store=store,
            provider=_Provider("text-embedding-3-small"),  # type: ignore[arg-type]
            mode="bm25",
            top_k=3,
        )
    assert result.hits


def test_empty_collection_accepts_any_model(tmp_path: Path) -> None:
    with SQLiteIndexStore(tmp_path / "crag.sqlite3") as store:
        result = retrieve_with_parents(
            "anything",
            store=store,
            provider=_Provider("some-new-model"),  # type: ignore[arg-type]
            mode="vector",
            top_k=3,
        )
    assert result.hits == []


def test_signature_cache_is_invalidated_by_upsert(tmp_path: Path) -> None:
    with SQLiteIndexStore(tmp_path / "crag.sqlite3") as store:
        assert store.embedding_signatures() == []
        _seed(store, "all-MiniLM-L6-v2")
        assert store.embedding_signatures() == [("mock", "all-MiniLM-L6-v2", 8)]
