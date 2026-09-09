"""Integration tests for index_file pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.ingestion import index_file
from custom_rag.storage import SQLiteIndexStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize("filename", ["sample.md", "sample.txt"])
def test_index_file_persists_fixture(tmp_path: Path, filename: str) -> None:
    store = SQLiteIndexStore(tmp_path / "crag.sqlite3")
    embedded = index_file(
        FIXTURES / filename,
        provider=MockEmbeddingProvider(),
        store=store,
        collection="default",
    )
    assert embedded.skipped is False
    assert embedded.children
    assert store.get_content_hash(embedded.metadata.source_path) == embedded.metadata.content_hash
    hits = store.search_vector(embedded.children[0].embedding, top_k=3)
    assert hits
    assert hits[0].parent is None
    if hits[0].chunk.parent_chunk_id is not None:
        parent = store.get_parent(hits[0].doc_id, hits[0].chunk.parent_chunk_id)
        assert parent is not None
    store.close()
