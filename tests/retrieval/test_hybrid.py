"""Tests for hybrid retrieve modes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from custom_rag.core.providers.mock import MockEmbeddingProvider
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


def _upsert_child(
    store: SQLiteIndexStore,
    *,
    doc_id: str,
    source_path: str,
    text: str,
    embedding: list[float],
    collection: str = "default",
) -> None:
    parent = Chunk(
        chunk_id="parent_root",
        role=ChunkRole.PARENT,
        text=f"Parent context: {text}",
        source_block_id="root",
        block_type=BlockType.DOCUMENT,
    )
    child = Chunk(
        chunk_id="child_0",
        role=ChunkRole.CHILD,
        text=text,
        source_block_id="p_0",
        parent_chunk_id="parent_root",
        block_type=BlockType.PARAGRAPH,
    )
    store.upsert(
        EmbeddedDocument(
            metadata=DocumentMetadata(
                doc_id=doc_id,
                source_path=source_path,
                source_uri=f"file://{source_path}",
                doc_type="text",
                content_hash=f"hash-{doc_id}",
                file_size=10,
                modified_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            parents=[parent],
            children=[
                EmbeddedChunk(
                    chunk=child,
                    embedding=embedding,
                    model="mock-embed-v1",
                    provider="mock",
                    dimensions=len(embedding),
                    embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                    text_hash=f"th-{doc_id}",
                )
            ],
        ),
        collection=collection,
    )


class _FixedProvider:
    """Returns a fixed query embedding for hybrid/vector tests.

    The model name matches what ``_upsert_child`` records, so the index-signature
    check treats these queries as coming from the model the index was built with.
    """

    name = "mock"
    model = "mock-embed-v1"
    dimensions = 8

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        return [list(self._vector)]


def test_hybrid_empty_bm25_falls_back_to_dense(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _upsert_child(
        store,
        doc_id="doc-1",
        source_path="a.txt",
        text="semantic only content without rare tokens",
        embedding=query_vec,
    )
    provider = _FixedProvider(query_vec)

    vector_result = retrieve_with_parents(
        "!!!",
        store=store,
        provider=provider,  # type: ignore[arg-type]
        mode="vector",
        top_k=3,
    )
    hybrid_result = retrieve_with_parents(
        "!!!",
        store=store,
        provider=provider,  # type: ignore[arg-type]
        mode="hybrid",
        top_k=3,
        candidate_k=10,
    )
    assert [hit.chunk_id for hit in hybrid_result.hits] == [
        hit.chunk_id for hit in vector_result.hits
    ]
    assert hybrid_result.hits[0].parent is not None
    store.close()


def test_hybrid_lexical_hit_beats_dense_distractor(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    distractor = [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lexical = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    _upsert_child(
        store,
        doc_id="doc-distract",
        source_path="distract.txt",
        text="completely unrelated filler prose",
        embedding=distractor,
    )
    _upsert_child(
        store,
        doc_id="doc-lex",
        source_path="lex.txt",
        text="contains UniqueKeywordAlpha for lexical match",
        embedding=lexical,
    )
    provider = _FixedProvider(query_vec)

    vector_result = retrieve_with_parents(
        "UniqueKeywordAlpha",
        store=store,
        provider=provider,  # type: ignore[arg-type]
        mode="vector",
        top_k=1,
    )
    assert vector_result.hits[0].doc_id == "doc-distract"

    hybrid_result = retrieve_with_parents(
        "UniqueKeywordAlpha",
        store=store,
        provider=provider,  # type: ignore[arg-type]
        mode="hybrid",
        top_k=1,
        candidate_k=10,
    )
    assert hybrid_result.hits[0].doc_id == "doc-lex"
    assert hybrid_result.hits[0].parent is not None
    assert "UniqueKeywordAlpha" in hybrid_result.hits[0].chunk.text
    assert "UniqueKeywordAlpha" in hybrid_result.hits[0].parent.text
    store.close()


def test_bm25_mode_attaches_parent(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    _upsert_child(
        store,
        doc_id="doc-1",
        source_path="a.txt",
        text="hello UniqueKeyword world",
        embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    result = retrieve_with_parents(
        "UniqueKeyword",
        store=store,
        provider=MockEmbeddingProvider(),
        mode="bm25",
        top_k=1,
    )
    assert result.hits
    assert result.hits[0].parent is not None
    assert result.mode == "bm25"
    store.close()
