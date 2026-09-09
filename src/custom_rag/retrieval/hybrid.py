"""Rank fusion helpers for hybrid retrieval."""

from __future__ import annotations


def rrf_fuse(
    ranked_lists: list[list[tuple[str, str]]],
    *,
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Fuse ranked (doc_id, chunk_id) lists with Reciprocal Rank Fusion.

    Ranks are 1-based. Identity is ``(doc_id, chunk_id)`` only.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")

    scores: dict[tuple[str, str], float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_id, chunk_id, score) for (doc_id, chunk_id), score in ordered[:top_k]]
