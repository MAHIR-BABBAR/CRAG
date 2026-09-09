"""Lightweight retrieval quality signal for agent consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from custom_rag.core.config import RetrievalModeName
from custom_rag.retrieval.packer import PackedContext
from custom_rag.storage.base import VectorHit

RetrievalQualityLabel = Literal["empty", "low", "high"]


@dataclass(frozen=True)
class QualityStats:
    hit_count: int
    chars_used: int
    truncated: bool
    top_score: float | None = None


@dataclass(frozen=True)
class RetrievalQuality:
    label: RetrievalQualityLabel
    stats: QualityStats


def score_retrieval(
    hits: list[VectorHit],
    packed: PackedContext,
    *,
    mode: RetrievalModeName,
    min_score_high: float = 0.25,
    min_bm25_high: float = 1.0,
    min_rrf_high: float = 0.025,
    reranked: bool = False,
) -> RetrievalQuality:
    """Classify retrieval as empty / low / high so an agent can branch on it.

    Each retrieval path scores on its own scale, so each gets its own threshold:

    - ``vector`` and cross-encoder reranked results are compared against
      ``min_score_high`` (cosine similarity, or the cross-encoder logit).
    - ``bm25`` returns SQLite's bm25() score, which is negative and better the
      more negative it is; its magnitude is compared against ``min_bm25_high``.
    - ``hybrid`` returns an RRF score. At the default ``rrf_k`` of 60, a passage
      ranked first by a single channel scores 1/61 ≈ 0.016, and one ranked
      highly by both scores roughly double that. ``min_rrf_high`` of 0.025 is
      therefore the point where the two channels agree rather than one guessing.
    """
    stats = QualityStats(
        hit_count=len(hits),
        chars_used=packed.chars_used,
        truncated=packed.truncated,
        top_score=hits[0].score if hits else None,
    )
    if not hits or stats.top_score is None:
        return RetrievalQuality(label="empty", stats=stats)
    if packed.chars_used == 0:
        return RetrievalQuality(label="empty", stats=stats)

    if reranked or mode == "vector":
        threshold = min_score_high
        value = stats.top_score
    elif mode == "bm25":
        threshold = min_bm25_high
        value = abs(stats.top_score)
    else:
        threshold = min_rrf_high
        value = stats.top_score

    label: RetrievalQualityLabel = "high" if value >= threshold else "low"
    return RetrievalQuality(label=label, stats=stats)
