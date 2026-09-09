"""Concurrency smoke for SQLiteIndexStore locking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from custom_rag.core.types import (
    BlockType,
    Chunk,
    ChunkRole,
    DocumentMetadata,
    EmbeddedChunk,
    EmbeddedDocument,
)
from custom_rag.storage.sqlite_store import SQLiteIndexStore


def _doc(doc_id: str, source_path: str, text: str) -> EmbeddedDocument:
    parent = Chunk(
        chunk_id="parent_root",
        role=ChunkRole.PARENT,
        text=f"Parent {text}",
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
    return EmbeddedDocument(
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
                embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                model="mock-embed-v1",
                provider="mock",
                dimensions=8,
                embedded_at=datetime(2026, 1, 1, tzinfo=UTC),
                text_hash=f"th-{doc_id}",
            )
        ],
    )


def test_concurrent_upsert_and_search(tmp_path: Path) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    store.upsert(_doc("seed", "seed.txt", "seed UniqueKeyword text"))

    def writer(index: int) -> None:
        store.upsert(_doc(f"doc-{index}", f"file-{index}.txt", f"UniqueKeyword writer {index}"))

    def reader() -> list:
        return store.search_bm25("UniqueKeyword", top_k=5)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(writer, i) for i in range(6)] + [
            pool.submit(reader) for _ in range(6)
        ]
        for future in futures:
            future.result()

    hits = store.search_bm25("UniqueKeyword", top_k=20)
    assert hits
    store.ping()
    store.close()
