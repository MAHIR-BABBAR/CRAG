"""Optional cross-encoder reranking of retrieval candidates."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from custom_rag.storage.base import VectorHit

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for rerank; "
            'install with: pip install "custom-rag[embeddings]"'
        ) from exc
    return CrossEncoder(model_name)


def rerank_hits(
    query: str,
    hits: list[VectorHit],
    *,
    model_name: str = DEFAULT_RERANK_MODEL,
    top_k: int | None = None,
) -> list[VectorHit]:
    """Score (query, passage) pairs with a cross-encoder and return top_k.

    Passage text prefers parent context when present, else child text.
    """
    if not hits:
        return []
    limit = len(hits) if top_k is None else max(1, top_k)
    model = _load_cross_encoder(model_name)
    pairs: list[list[str]] = []
    for hit in hits:
        passage = hit.parent.text if hit.parent is not None else hit.chunk.text
        pairs.append([query, passage])
    scores = model.predict(pairs)
    ranked = sorted(
        zip(hits, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    out: list[VectorHit] = []
    for hit, score in ranked[:limit]:
        out.append(
            VectorHit(
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                score=float(score),
                chunk=hit.chunk,
                parent=hit.parent,
            )
        )
    return out
