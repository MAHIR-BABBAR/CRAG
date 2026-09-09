"""The numpy dense-search path must agree with the pure-Python one, exactly."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from custom_rag.core.types import (
    BlockType,
    Chunk,
    ChunkRole,
    DocumentMetadata,
    EmbeddedChunk,
    EmbeddedDocument,
)
from custom_rag.storage.sqlite_store import SQLiteIndexStore

BACKENDS = ("python", "numpy")


def _doc(doc_id: str, embedding: list[float], text: str) -> EmbeddedDocument:
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
            source_path=f"{doc_id}.txt",
            source_uri=f"file://{doc_id}.txt",
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
    )


def _store(tmp_path: Path, backend: str, name: str = "index") -> SQLiteIndexStore:
    return SQLiteIndexStore(tmp_path / f"{name}-{backend}.sqlite3", dense_search=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_returns_nearest_document(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("numpy")
    with _store(tmp_path, backend) as store:
        store.upsert(_doc("a", [1.0, 0.0, 0.0], "alpha"))
        store.upsert(_doc("b", [0.0, 1.0, 0.0], "beta"))
        hits = store.search_vector([0.9, 0.1, 0.0], top_k=1)
        assert [hit.doc_id for hit in hits] == ["a"]


def test_backends_agree_on_full_ranking_over_random_vectors(tmp_path: Path) -> None:
    """Same corpus, same query, same order and scores — this is not an ANN index."""
    pytest.importorskip("numpy")
    rng = random.Random(20260909)
    corpus = {f"doc-{i}": [rng.uniform(-1.0, 1.0) for _ in range(16)] for i in range(50)}
    query = [rng.uniform(-1.0, 1.0) for _ in range(16)]

    rankings: dict[str, list[tuple[str, float]]] = {}
    for backend in BACKENDS:
        with _store(tmp_path, backend, name="parity") as store:
            for doc_id, vector in corpus.items():
                store.upsert(_doc(doc_id, vector, f"text for {doc_id}"))
            hits = store.search_vector(query, top_k=len(corpus))
            rankings[backend] = [(hit.doc_id, hit.score) for hit in hits]

    python_ids = [doc_id for doc_id, _ in rankings["python"]]
    numpy_ids = [doc_id for doc_id, _ in rankings["numpy"]]
    assert python_ids == numpy_ids
    for (_, python_score), (_, numpy_score) in zip(
        rankings["python"], rankings["numpy"], strict=True
    ):
        assert python_score == pytest.approx(numpy_score, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_collection_returns_no_hits(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("numpy")
    with _store(tmp_path, backend, name="empty") as store:
        assert store.search_vector([1.0, 0.0, 0.0], top_k=5) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_query_vector_yields_no_ranking(tmp_path: Path, backend: str) -> None:
    """A zero vector has no direction, so it has no nearest neighbour."""
    pytest.importorskip("numpy")
    with _store(tmp_path, backend, name="zeroq") as store:
        store.upsert(_doc("a", [1.0, 0.0, 0.0], "alpha"))
        hits = store.search_vector([0.0, 0.0, 0.0], top_k=5)
        assert all(hit.score == 0.0 for hit in hits)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_stored_vector_scores_zero(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("numpy")
    with _store(tmp_path, backend, name="zerodoc") as store:
        store.upsert(_doc("zero", [0.0, 0.0, 0.0], "zero"))
        store.upsert(_doc("real", [1.0, 0.0, 0.0], "real"))
        hits = store.search_vector([1.0, 0.0, 0.0], top_k=2)
        scores = {hit.doc_id: hit.score for hit in hits}
        assert scores["real"] == pytest.approx(1.0)
        assert scores["zero"] == pytest.approx(0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_top_k_larger_than_corpus_is_clamped(tmp_path: Path, backend: str) -> None:
    pytest.importorskip("numpy")
    with _store(tmp_path, backend, name="clamp") as store:
        store.upsert(_doc("a", [1.0, 0.0, 0.0], "alpha"))
        store.upsert(_doc("b", [0.0, 1.0, 0.0], "beta"))
        assert len(store.search_vector([1.0, 0.0, 0.0], top_k=99)) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_mismatched_dimensions_are_skipped(tmp_path: Path, backend: str) -> None:
    """A stale 3-d vector must not crash a 4-d query, in either backend."""
    pytest.importorskip("numpy")
    with _store(tmp_path, backend, name="dims") as store:
        store.upsert(_doc("short", [1.0, 0.0, 0.0], "short"))
        store.upsert(_doc("long", [1.0, 0.0, 0.0, 0.0], "long"))
        hits = store.search_vector([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert [hit.doc_id for hit in hits] == ["long"]


def test_unknown_backend_falls_back_to_python(tmp_path: Path) -> None:
    with SQLiteIndexStore(tmp_path / "unknown.sqlite3", dense_search="hnsw") as store:
        assert store.dense_search == "python"
