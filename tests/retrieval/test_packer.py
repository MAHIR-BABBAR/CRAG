"""Tests for char-budget context packing."""

from __future__ import annotations

from custom_rag.core.types import BlockType, Chunk, ChunkRole
from custom_rag.retrieval.packer import assemble_context
from custom_rag.storage.base import VectorHit


def _hit(
    *,
    doc_id: str = "doc-1",
    chunk_id: str = "child_0",
    text: str = "child text",
    parent_text: str | None = "parent text",
    parent_id: str | None = "parent_root",
    score: float = 0.9,
) -> VectorHit:
    child = Chunk(
        chunk_id=chunk_id,
        role=ChunkRole.CHILD,
        text=text,
        source_block_id="p_0",
        parent_chunk_id=parent_id,
        block_type=BlockType.PARAGRAPH,
    )
    parent = None
    if parent_text is not None and parent_id is not None:
        parent = Chunk(
            chunk_id=parent_id,
            role=ChunkRole.PARENT,
            text=parent_text,
            source_block_id="root",
            block_type=BlockType.DOCUMENT,
        )
    return VectorHit(
        doc_id=doc_id,
        chunk_id=chunk_id,
        score=score,
        chunk=child,
        parent=parent,
    )


def test_assemble_empty_hits() -> None:
    packed = assemble_context([], max_chars=100)
    assert packed.text == ""
    assert packed.citations == []
    assert packed.truncated is False
    assert packed.chars_used == 0


def test_assemble_prefers_parent_and_respects_budget() -> None:
    hits = [
        _hit(parent_text="AAAA" * 50, text="child"),
    ]
    packed = assemble_context(
        hits,
        max_chars=40,
        include_parents=True,
        source_path_for=lambda _doc_id: "a.txt",
    )
    assert len(packed.text) <= 40
    assert packed.truncated is True
    assert packed.citations
    assert packed.citations[0].source_path == "a.txt"
    assert packed.citations[0].index == 1
    assert "[1]" in packed.text


def test_assemble_dedupes_same_parent() -> None:
    hits = [
        _hit(chunk_id="c1", text="c1", parent_text="shared parent", score=0.9),
        _hit(chunk_id="c2", text="c2", parent_text="shared parent", score=0.8),
    ]
    packed = assemble_context(
        hits,
        max_chars=4000,
        include_parents=True,
        source_path_for=lambda _doc_id: "a.txt",
    )
    assert len(packed.citations) == 1
    assert packed.text.count("[1]") == 1


def test_assemble_falls_back_to_child_without_parent() -> None:
    hits = [_hit(parent_text=None, parent_id=None, text="only child body")]
    packed = assemble_context(hits, max_chars=4000, include_parents=True)
    assert "only child body" in packed.text
