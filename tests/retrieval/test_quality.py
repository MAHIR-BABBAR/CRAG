"""Tests for retrieval quality labels."""

from __future__ import annotations

from custom_rag.core.types import BlockType, Chunk, ChunkRole
from custom_rag.retrieval.packer import PackedContext, assemble_context
from custom_rag.retrieval.quality import score_retrieval
from custom_rag.storage.base import VectorHit


def _hit(score: float) -> VectorHit:
    child = Chunk(
        chunk_id="child_0",
        role=ChunkRole.CHILD,
        text="child",
        source_block_id="p_0",
        parent_chunk_id="parent_root",
        block_type=BlockType.PARAGRAPH,
    )
    return VectorHit(
        doc_id="doc-1",
        chunk_id="child_0",
        score=score,
        chunk=child,
        parent=None,
    )


def test_quality_empty() -> None:
    packed = PackedContext(text="", citations=[], truncated=False, chars_used=0)
    quality = score_retrieval([], packed, mode="vector", min_score_high=0.25)
    assert quality.label == "empty"
    assert quality.stats.hit_count == 0


def test_quality_high_for_vector_score() -> None:
    hits = [_hit(0.9)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="vector", min_score_high=0.25)
    assert quality.label == "high"


def test_quality_low_for_weak_vector_score() -> None:
    hits = [_hit(0.1)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="vector", min_score_high=0.25)
    assert quality.label == "low"


def test_hybrid_single_channel_hit_is_low() -> None:
    """1/61 is what a passage scores when only one channel ranked it first."""
    hits = [_hit(1 / 61)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="hybrid", min_rrf_high=0.025)
    assert quality.label == "low"


def test_hybrid_two_channel_agreement_is_high() -> None:
    """Ranked first by both BM25 and dense: 2/61, the signal we call high."""
    hits = [_hit(2 / 61)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="hybrid", min_rrf_high=0.025)
    assert quality.label == "high"


def test_quality_low_for_weak_reranked_hybrid() -> None:
    hits = [_hit(0.1)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="hybrid", min_score_high=0.25, reranked=True)
    assert quality.label == "low"


def test_strong_bm25_hit_is_high() -> None:
    """SQLite bm25() is negative and better the further from zero it gets."""
    hits = [_hit(-2.5)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="bm25", min_bm25_high=1.0)
    assert quality.label == "high"


def test_marginal_bm25_hit_is_low() -> None:
    hits = [_hit(-0.2)]
    packed = assemble_context(hits, max_chars=4000, include_parents=False)
    quality = score_retrieval(hits, packed, mode="bm25", min_bm25_high=1.0)
    assert quality.label == "low"


def test_hits_that_pack_to_nothing_are_empty() -> None:
    """A hit that contributes no characters is not context an agent can use."""
    hits = [_hit(0.9)]
    packed = PackedContext(text="", citations=[], truncated=True, chars_used=0)
    quality = score_retrieval(hits, packed, mode="vector", min_score_high=0.25)
    assert quality.label == "empty"
