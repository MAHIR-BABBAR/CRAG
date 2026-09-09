"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from custom_rag.retrieval.hybrid import rrf_fuse


def test_rrf_dual_list_id_outranks_single_list() -> None:
    dense = [("doc", "both"), ("doc", "dense_only")]
    bm25 = [("doc", "both"), ("doc", "bm25_only")]
    fused = rrf_fuse([dense, bm25], rrf_k=60, top_k=3)
    assert fused[0][0] == "doc"
    assert fused[0][1] == "both"
    assert fused[0][2] > fused[1][2]


def test_rrf_uses_ranks_not_scores() -> None:
    # Same identity order → identical RRF contribution regardless of "score" semantics.
    left = [("a", "1"), ("a", "2")]
    right = [("a", "2"), ("a", "1")]
    fused = rrf_fuse([left, right], rrf_k=60, top_k=2)
    scores = {chunk_id: score for _, chunk_id, score in fused}
    assert scores["1"] == pytest.approx(scores["2"])


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([[("d", "c")]], rrf_k=0, top_k=1)
    with pytest.raises(ValueError):
        rrf_fuse([[("d", "c")]], rrf_k=60, top_k=0)
