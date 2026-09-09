"""Pydantic request/response models for the open socket API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from custom_rag.core.config import HARD_MAX_CANDIDATE_K, HARD_MAX_TOP_K

EmbeddingProviderName = Literal["openai", "local", "mock"]
RetrievalModeName = Literal["vector", "bm25", "hybrid"]


class HealthResponse(BaseModel):
    status: str
    version: str
    storage_path: str
    collection: str


class IndexRequest(BaseModel):
    path: str
    collection: str | None = None
    provider: EmbeddingProviderName | None = None
    recursive: bool = True


class IndexResponse(BaseModel):
    """Single-file index result (backward compatible)."""

    skipped: bool
    doc_id: str
    source_path: str
    children: int
    parents: int
    collection: str


class IndexBatchResponse(BaseModel):
    """Aggregate result when indexing a directory."""

    indexed: int
    skipped: int
    failed: list[dict[str, str]]
    file_count: int
    truncated: bool = Field(
        default=False,
        description="True when the directory held more indexable files than max_files.",
    )
    max_files: int
    collection: str
    path: str


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: RetrievalModeName | None = None
    top_k: int | None = Field(default=None, ge=1, le=HARD_MAX_TOP_K)
    candidate_k: int | None = Field(default=None, ge=1, le=HARD_MAX_CANDIDATE_K)
    collection: str | None = None
    provider: EmbeddingProviderName | None = None
    rerank: bool | None = None


class RetrieveResponse(BaseModel):
    query: str
    mode: RetrievalModeName
    hits: list[dict[str, Any]]
    reranked: bool = False


class ContextRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: RetrievalModeName | None = None
    top_k: int | None = Field(default=None, ge=1, le=HARD_MAX_TOP_K)
    candidate_k: int | None = Field(default=None, ge=1, le=HARD_MAX_CANDIDATE_K)
    collection: str | None = None
    provider: EmbeddingProviderName | None = None
    max_chars: int | None = Field(default=None, ge=1)
    rerank: bool | None = None


class CitationModel(BaseModel):
    index: int
    doc_id: str
    chunk_id: str
    parent_chunk_id: str | None = None
    source_path: str
    score: float
    page: int | None = None
    slide: int | None = None


class QualityStatsModel(BaseModel):
    hit_count: int
    chars_used: int
    truncated: bool
    top_score: float | None = None


class ContextResponse(BaseModel):
    query: str
    mode: RetrievalModeName
    context: str
    citations: list[CitationModel]
    retrieval_quality: Literal["empty", "low", "high"]
    quality_stats: QualityStatsModel
    hits: list[dict[str, Any]]
