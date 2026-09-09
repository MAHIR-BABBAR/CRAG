"""Retrieval helpers: channel search, hybrid RRF, optional CE rerank, parent attach."""

from __future__ import annotations

from dataclasses import dataclass

from custom_rag.core.config import CRAGSettings, RetrievalModeName, get_settings
from custom_rag.core.exceptions import ConfigError
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.registry import resolve_embedding_provider
from custom_rag.core.types import Chunk
from custom_rag.retrieval.hybrid import rrf_fuse
from custom_rag.storage.base import IndexStore, VectorHit
from custom_rag.storage.sqlite_store import SQLiteIndexStore


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    hits: list[VectorHit]
    mode: RetrievalModeName = "vector"
    reranked: bool = False

    def as_dicts(self) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for hit in self.hits:
            payload.append(
                {
                    "doc_id": hit.doc_id,
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "child": hit.chunk.model_dump(mode="json"),
                    "parent": None if hit.parent is None else hit.parent.model_dump(mode="json"),
                }
            )
        return payload


def retrieve_with_parents(
    query: str,
    *,
    store: IndexStore | None = None,
    provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    collection: str | None = None,
    mode: RetrievalModeName | None = None,
    rrf_k: int | None = None,
    settings: CRAGSettings | None = None,
    rerank: bool | None = None,
) -> RetrievalResult:
    """Search child chunks and attach owning parent chunks.

    Modes:
    - ``vector``: dense cosine over child embeddings
    - ``bm25``: FTS5 lexical search
    - ``hybrid``: dense ∥ BM25 → RRF → top_k, then attach parents once

    When ``rerank`` is enabled, candidates (up to ``rerank_top_n``) are scored
    with a cross-encoder before cutting to ``top_k``.
    """
    active_settings = settings or get_settings()
    active_store = store or SQLiteIndexStore(
        active_settings.storage.path,
        default_collection=active_settings.storage.collection,
        dense_search=active_settings.storage.dense_search,
    )
    active_mode: RetrievalModeName = mode or active_settings.retrieval.mode
    active_top_k = top_k or active_settings.retrieval.top_k
    active_candidate_k = candidate_k or active_settings.retrieval.candidate_k
    active_rrf_k = rrf_k or active_settings.retrieval.rrf_k
    active_collection = collection or active_settings.storage.collection
    do_rerank = active_settings.retrieval.rerank_enabled if rerank is None else rerank
    rerank_top_n = max(active_settings.retrieval.rerank_top_n, active_top_k)

    if active_mode == "vector":
        active_provider = provider or resolve_embedding_provider(active_settings)
        _assert_query_model_matches(active_store, active_provider, active_collection)
        query_embedding = active_provider.embed_texts([query])[0]
        pool_k = rerank_top_n if do_rerank else active_top_k
        hits = active_store.search_vector(
            query_embedding,
            top_k=pool_k,
            collection=active_collection,
        )
        attached = _attach_parents(active_store, hits)
        return _maybe_rerank(
            query,
            attached,
            mode=active_mode,
            top_k=active_top_k,
            do_rerank=do_rerank,
            model_name=active_settings.retrieval.rerank_model,
        )

    if active_mode == "bm25":
        pool_k = rerank_top_n if do_rerank else active_top_k
        hits = active_store.search_bm25(
            query,
            top_k=pool_k,
            collection=active_collection,
        )
        attached = _attach_parents(active_store, hits)
        return _maybe_rerank(
            query,
            attached,
            mode=active_mode,
            top_k=active_top_k,
            do_rerank=do_rerank,
            model_name=active_settings.retrieval.rerank_model,
        )

    # hybrid
    active_provider = provider or resolve_embedding_provider(active_settings)
    _assert_query_model_matches(active_store, active_provider, active_collection)
    query_embedding = active_provider.embed_texts([query])[0]
    channel_k = max(active_candidate_k, rerank_top_n if do_rerank else active_candidate_k)
    dense_hits = active_store.search_vector(
        query_embedding,
        top_k=channel_k,
        collection=active_collection,
    )
    bm25_hits = active_store.search_bm25(
        query,
        top_k=channel_k,
        collection=active_collection,
    )

    if not bm25_hits:
        attached = _attach_parents(active_store, dense_hits[: max(active_top_k, rerank_top_n)])
        return _maybe_rerank(
            query,
            attached,
            mode=active_mode,
            top_k=active_top_k,
            do_rerank=do_rerank,
            model_name=active_settings.retrieval.rerank_model,
        )

    fuse_k = rerank_top_n if do_rerank else active_top_k
    dense_keys = [(hit.doc_id, hit.chunk_id) for hit in dense_hits]
    bm25_keys = [(hit.doc_id, hit.chunk_id) for hit in bm25_hits]
    fused = rrf_fuse(
        [dense_keys, bm25_keys],
        rrf_k=active_rrf_k,
        top_k=fuse_k,
    )

    by_key = {(hit.doc_id, hit.chunk_id): hit for hit in dense_hits}
    for hit in bm25_hits:
        by_key.setdefault((hit.doc_id, hit.chunk_id), hit)

    fused_hits: list[VectorHit] = []
    for doc_id, chunk_id, score in fused:
        base = by_key[(doc_id, chunk_id)]
        fused_hits.append(
            VectorHit(
                doc_id=base.doc_id,
                chunk_id=base.chunk_id,
                score=score,
                chunk=base.chunk,
                parent=None,
            )
        )

    attached = _attach_parents(active_store, fused_hits)
    return _maybe_rerank(
        query,
        attached,
        mode=active_mode,
        top_k=active_top_k,
        do_rerank=do_rerank,
        model_name=active_settings.retrieval.rerank_model,
    )


def _maybe_rerank(
    query: str,
    hits: list[VectorHit],
    *,
    mode: RetrievalModeName,
    top_k: int,
    do_rerank: bool,
    model_name: str,
) -> RetrievalResult:
    if not do_rerank:
        return RetrievalResult(query=query, hits=hits[:top_k], mode=mode, reranked=False)
    from custom_rag.retrieval.rerank import rerank_hits

    ranked = rerank_hits(query, hits, model_name=model_name, top_k=top_k)
    return RetrievalResult(query=query, hits=ranked, mode=mode, reranked=True)


def _assert_query_model_matches(
    store: IndexStore,
    provider: EmbeddingProvider,
    collection: str,
) -> None:
    """Refuse to compare a query vector against a differently-embedded index.

    Cosine similarity between two embedding spaces is noise that still ranks, so
    a mismatch fails silently as bad results rather than as an error.
    """
    signatures = store.embedding_signatures(collection)
    if not signatures:
        return
    if any(model == provider.model for _provider, model, _dim in signatures):
        return
    indexed = ", ".join(sorted({f"{model} ({dim}d)" for _p, model, dim in signatures}))
    raise ConfigError(
        f"collection {collection!r} was indexed with {indexed}, but the query was "
        f"embedded with {provider.model!r} ({provider.dimensions}d). Re-index the "
        f"collection or query with the model it was built from."
    )


def _attach_parents(store: IndexStore, hits: list[VectorHit]) -> list[VectorHit]:
    attached: list[VectorHit] = []
    for hit in hits:
        parent = None
        if hit.chunk.parent_chunk_id:
            parent = store.get_parent(hit.doc_id, hit.chunk.parent_chunk_id)
        attached.append(
            VectorHit(
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                score=hit.score,
                chunk=hit.chunk,
                parent=parent,
            )
        )
    return attached


def format_hit_preview(hit: VectorHit, *, width: int = 80) -> tuple[str, str]:
    child_preview = hit.chunk.text[:width].replace("\n", " ")
    parent_text = hit.parent.text if isinstance(hit.parent, Chunk) and hit.parent else ""
    parent_preview = parent_text[:width].replace("\n", " ") if parent_text else "-"
    return child_preview, parent_preview
