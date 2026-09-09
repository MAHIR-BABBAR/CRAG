"""Unit tests for IR metrics."""

from __future__ import annotations

import pytest

from custom_rag.eval.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    summarize_metrics,
)


def test_hit_at_k() -> None:
    ranks = [1, None, 3, 2]
    assert hit_at_k(ranks, k=2) == 0.5
    assert hit_at_k(ranks, k=3) == 0.75


def test_mrr() -> None:
    ranks = [1, 2, None]
    assert mean_reciprocal_rank(ranks) == (1.0 + 0.5 + 0.0) / 3


def test_ndcg_at_k_perfect_and_miss() -> None:
    from custom_rag.eval.metrics import ndcg_at_k

    perfect = ndcg_at_k([[1.0, 0.0, 0.0]], [1], k=3)
    assert perfect == pytest.approx(1.0)
    miss = ndcg_at_k([[0.0, 0.0, 0.0]], [1], k=3)
    assert miss == 0.0
    empty_gold = ndcg_at_k([[0.0, 0.0]], [0], k=2)
    assert empty_gold == 1.0


def test_recall_at_k_and_summary() -> None:
    recall = recall_at_k([1, 0, 0], [1, 2, 0], k=5)
    assert recall == (1.0 + 0.0 + 1.0) / 3
    summary = summarize_metrics(
        [1, None, 1],
        [1, 0, 0],
        [1, 2, 0],
        k=5,
        relevance_lists=[[1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0]],
    )
    assert summary.hit_at_k == hit_at_k([1, None, 1], k=5)
    assert 0.0 <= summary.ndcg_at_k <= 1.0
    assert summary.queries == 3


def test_ndcg_clamps_when_more_hits_than_gold() -> None:
    from custom_rag.eval.metrics import ndcg_at_k

    # Two binary hits but gold=1 must not exceed 1.0.
    score = ndcg_at_k([[1.0, 1.0, 0.0]], [1], k=3)
    assert score == pytest.approx(1.0)
