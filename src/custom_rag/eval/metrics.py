"""Retrieval evaluation metrics (IR-style)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}


@dataclass(frozen=True)
class QueryMetrics:
    """Metrics for a single query, kept per-query so they can be resampled."""

    hit: float
    recall: float
    reciprocal_rank: float
    ndcg: float


@dataclass(frozen=True)
class MetricResult:
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    queries: int
    intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)


def hit_at_k(relevant_ranks: list[int | None], *, k: int) -> float:
    """Fraction of queries with at least one relevant hit in top-k.

    ``relevant_ranks`` entries are 1-based ranks of the first relevant hit,
    or ``None`` when no relevant hit appears in the ranked list.
    """
    if not relevant_ranks:
        return 0.0
    hits = sum(1 for rank in relevant_ranks if rank is not None and rank <= k)
    return hits / len(relevant_ranks)


def recall_at_k(
    retrieved_relevant_counts: list[int],
    gold_counts: list[int],
    *,
    k: int | None = None,
) -> float:
    """Mean recall: relevant_in_top_k / gold_relevant (capped at 1.0 per query)."""
    _ = k
    if not retrieved_relevant_counts:
        return 0.0
    if len(retrieved_relevant_counts) != len(gold_counts):
        raise ValueError("retrieved_relevant_counts and gold_counts length mismatch")
    total = 0.0
    for found, gold in zip(retrieved_relevant_counts, gold_counts, strict=True):
        if gold <= 0:
            total += 1.0 if found == 0 else 0.0
        else:
            total += min(1.0, found / gold)
    return total / len(retrieved_relevant_counts)


def mean_reciprocal_rank(relevant_ranks: list[int | None]) -> float:
    """Mean reciprocal rank using 1-based first-relevant ranks."""
    if not relevant_ranks:
        return 0.0
    total = 0.0
    for rank in relevant_ranks:
        if rank is not None and rank >= 1:
            total += 1.0 / rank
    return total / len(relevant_ranks)


def _dcg(relevances: list[float], *, k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        total += float(rel) / math.log2(index + 1)
    return total


def ndcg_at_k(
    relevance_lists: list[list[float]],
    gold_counts: list[int],
    *,
    k: int,
) -> float:
    """Mean nDCG@k with gold-aware ideal DCG (binary 0/1 relevances OK)."""
    if not relevance_lists:
        return 0.0
    if len(relevance_lists) != len(gold_counts):
        raise ValueError("relevance_lists and gold_counts length mismatch")
    scores: list[float] = []
    for rels, gold in zip(relevance_lists, gold_counts, strict=True):
        truncated = list(rels[:k]) + [0.0] * max(0, k - len(rels))
        truncated = truncated[:k]
        if gold <= 0:
            scores.append(1.0 if all(r == 0.0 for r in truncated) else 0.0)
            continue
        ideal = [1.0] * min(gold, k)
        idcg_val = _dcg(ideal, k=k)
        raw = 0.0 if idcg_val == 0.0 else _dcg(truncated, k=k) / idcg_val
        scores.append(min(1.0, raw))
    return sum(scores) / len(scores)


def evaluate_ranking(
    relevance: list[float],
    gold_count: int,
    *,
    k: int,
) -> QueryMetrics:
    """Score one query's ranked list.

    ``relevance`` is the graded relevance of each returned item in rank order.
    ``gold_count`` of zero means the query is expected to return nothing, and is
    scored as a perfect result when it does.
    """
    truncated = list(relevance[:k]) + [0.0] * max(0, k - len(relevance))
    truncated = truncated[:k]
    found = sum(1 for value in truncated if value > 0.0)
    first_rank = next(
        (index for index, value in enumerate(truncated, start=1) if value > 0.0), None
    )

    if gold_count <= 0:
        clean = found == 0
        return QueryMetrics(
            hit=1.0 if clean else 0.0,
            recall=1.0 if clean else 0.0,
            reciprocal_rank=1.0 if clean else 0.0,
            ndcg=1.0 if clean else 0.0,
        )

    idcg = _dcg([1.0] * min(gold_count, k), k=k)
    return QueryMetrics(
        hit=1.0 if first_rank is not None else 0.0,
        recall=min(1.0, found / gold_count),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        ndcg=0.0 if idcg == 0.0 else min(1.0, _dcg(truncated, k=k) / idcg),
    )


def bootstrap_ci(
    values: list[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260909,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for the mean of per-query scores.

    Resampling queries with replacement answers the question that matters when
    comparing two retrieval modes on a few hundred queries: how much of the gap
    is the method, and how much is which queries happened to be in the set.
    """
    if not values:
        return ConfidenceInterval(low=0.0, high=0.0)
    if len(values) == 1:
        return ConfidenceInterval(low=values[0], high=values[0])

    rng = random.Random(seed)
    size = len(values)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * iterations) - 1)
    high_index = min(iterations - 1, int((1.0 - tail) * iterations))
    return ConfidenceInterval(low=means[low_index], high=means[high_index])


def summarize_query_metrics(
    per_query: list[QueryMetrics],
    *,
    k: int,
    bootstrap_iterations: int = 1000,
) -> MetricResult:
    """Aggregate per-query scores into means with bootstrap intervals."""
    if not per_query:
        return MetricResult(hit_at_k=0.0, recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0, k=k, queries=0)

    columns = {
        "hit_at_k": [item.hit for item in per_query],
        "recall_at_k": [item.recall for item in per_query],
        "mrr": [item.reciprocal_rank for item in per_query],
        "ndcg_at_k": [item.ndcg for item in per_query],
    }
    means = {name: sum(values) / len(values) for name, values in columns.items()}
    intervals = {
        name: bootstrap_ci(values, iterations=bootstrap_iterations)
        for name, values in columns.items()
    }
    return MetricResult(
        hit_at_k=means["hit_at_k"],
        recall_at_k=means["recall_at_k"],
        mrr=means["mrr"],
        ndcg_at_k=means["ndcg_at_k"],
        k=k,
        queries=len(per_query),
        intervals=intervals,
    )


def summarize_metrics(
    relevant_ranks: list[int | None],
    retrieved_relevant_counts: list[int],
    gold_counts: list[int],
    *,
    k: int,
    relevance_lists: list[list[float]] | None = None,
) -> MetricResult:
    if relevance_lists is None:
        relevance_lists = []
        for rank in relevant_ranks:
            row = [0.0] * k
            if rank is not None and 1 <= rank <= k:
                row[rank - 1] = 1.0
            relevance_lists.append(row)

    return MetricResult(
        hit_at_k=hit_at_k(relevant_ranks, k=k),
        recall_at_k=recall_at_k(retrieved_relevant_counts, gold_counts, k=k),
        mrr=mean_reciprocal_rank(relevant_ranks),
        ndcg_at_k=ndcg_at_k(relevance_lists, gold_counts, k=k),
        k=k,
        queries=len(relevant_ranks),
    )
