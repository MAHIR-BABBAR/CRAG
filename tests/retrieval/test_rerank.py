"""Unit tests for optional cross-encoder rerank helper."""

from __future__ import annotations

from custom_rag.core.types import BlockType, Chunk, ChunkRole
from custom_rag.retrieval.rerank import rerank_hits
from custom_rag.storage.base import VectorHit


def _hit(chunk_id: str, text: str, score: float = 0.0) -> VectorHit:
    child = Chunk(
        chunk_id=chunk_id,
        role=ChunkRole.CHILD,
        text=text,
        source_block_id="p_0",
        parent_chunk_id=None,
        block_type=BlockType.PARAGRAPH,
    )
    return VectorHit(
        doc_id="doc-1",
        chunk_id=chunk_id,
        score=score,
        chunk=child,
        parent=None,
    )


def test_rerank_hits_empty() -> None:
    assert rerank_hits("q", []) == []


def test_rerank_hits_orders_by_mock_scores(monkeypatch) -> None:
    class FakeCE:
        def predict(self, pairs):
            # Prefer the passage containing "relevant"
            return [1.0 if "relevant" in pair[1] else -1.0 for pair in pairs]

    monkeypatch.setattr(
        "custom_rag.retrieval.rerank._load_cross_encoder",
        lambda _name: FakeCE(),
    )
    hits = [
        _hit("a", "unrelated noise", score=0.9),
        _hit("b", "this is relevant evidence", score=0.1),
    ]
    ranked = rerank_hits("claim", hits, top_k=1)
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "b"
    assert ranked[0].score == 1.0
