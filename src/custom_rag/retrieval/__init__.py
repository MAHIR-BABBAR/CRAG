"""Hybrid search, reranking, and context assembly."""

from custom_rag.retrieval.context import RetrievalResult, retrieve_with_parents
from custom_rag.retrieval.hybrid import rrf_fuse
from custom_rag.retrieval.packer import Citation, PackedContext, assemble_context
from custom_rag.retrieval.quality import RetrievalQuality, score_retrieval
from custom_rag.retrieval.rerank import DEFAULT_RERANK_MODEL, rerank_hits

__all__ = [
    "DEFAULT_RERANK_MODEL",
    "Citation",
    "PackedContext",
    "RetrievalQuality",
    "RetrievalResult",
    "assemble_context",
    "rerank_hits",
    "retrieve_with_parents",
    "rrf_fuse",
    "score_retrieval",
]
