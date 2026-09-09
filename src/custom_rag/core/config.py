"""Configuration loading from YAML and environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProviderName = Literal["openai", "local", "mock"]
StorageBackendName = Literal["sqlite"]
DenseSearchName = Literal["python", "numpy"]
RetrievalModeName = Literal["vector", "bm25", "hybrid"]

# Absolute request ceilings. Config values are validated against these so the
# API schema and the settings model cannot drift apart.
HARD_MAX_TOP_K = 100
HARD_MAX_CANDIDATE_K = 200

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


class EmbeddingSettings(BaseModel):
    provider: EmbeddingProviderName = "local"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, ge=1)
    dimensions: int | None = None


class StorageSettings(BaseModel):
    backend: StorageBackendName = "sqlite"
    path: str = "workspace_index/crag.sqlite3"
    collection: str = "default"
    dense_search: DenseSearchName = "python"


class RetrievalSettings(BaseModel):
    mode: RetrievalModeName = "hybrid"
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    max_top_k: int = Field(default=100, ge=1, le=HARD_MAX_TOP_K)
    max_candidate_k: int = Field(default=200, ge=1, le=HARD_MAX_CANDIDATE_K)
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = Field(
        default=20,
        ge=1,
        description="Candidates to score with the cross-encoder before cutting to top_k.",
    )


class ApiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: str | None = None
    max_index_files: int = Field(default=500, ge=1)
    allowed_index_roots: list[str] = Field(
        default_factory=list,
        description=(
            "Directories the /v1/index endpoint may read from. Empty means any "
            "path the server process can read, which is only safe on loopback."
        ),
    )


class ContextSettings(BaseModel):
    max_chars: int = Field(default=4000, ge=1)
    include_parents: bool = True
    # Per-channel thresholds for the `high` retrieval-quality label. Each channel
    # produces scores on a different scale, so one threshold cannot serve all.
    min_score_high: float = 0.25  # cosine similarity (vector) or cross-encoder logit
    min_bm25_high: float = 1.0  # magnitude of the (negative) SQLite bm25 score
    min_rrf_high: float = 0.025  # ~top-3 in both RRF channels at rrf_k=60


class CRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_provider: EmbeddingProviderName | None = Field(
        default=None, alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str | None = Field(default=None, alias="EMBEDDING_MODEL")
    embedding_batch_size: int | None = Field(default=None, alias="EMBEDDING_BATCH_SIZE", ge=1)
    crag_storage_path: str | None = Field(default=None, alias="CRAG_STORAGE_PATH")
    crag_collection: str | None = Field(default=None, alias="CRAG_COLLECTION")
    crag_retrieval_mode: RetrievalModeName | None = Field(default=None, alias="CRAG_RETRIEVAL_MODE")
    crag_top_k: int | None = Field(default=None, alias="CRAG_TOP_K", ge=1)
    crag_candidate_k: int | None = Field(default=None, alias="CRAG_CANDIDATE_K", ge=1)
    crag_api_host: str | None = Field(default=None, alias="CRAG_API_HOST")
    crag_api_port: int | None = Field(default=None, alias="CRAG_API_PORT", ge=1, le=65535)
    crag_context_max_chars: int | None = Field(default=None, alias="CRAG_CONTEXT_MAX_CHARS", ge=1)
    crag_api_key: str | None = Field(default=None, alias="CRAG_API_KEY")
    crag_dense_search: DenseSearchName | None = Field(default=None, alias="CRAG_DENSE_SEARCH")
    crag_allowed_index_roots: str | None = Field(default=None, alias="CRAG_ALLOWED_INDEX_ROOTS")

    def model_post_init(self, __context: Any) -> None:
        if self.embedding_provider is not None:
            self.embedding.provider = self.embedding_provider
        if self.embedding_model is not None:
            self.embedding.model = self.embedding_model
        if self.embedding_batch_size is not None:
            self.embedding.batch_size = self.embedding_batch_size
        if self.crag_storage_path is not None:
            self.storage.path = self.crag_storage_path
        if self.crag_collection is not None:
            self.storage.collection = self.crag_collection
        if self.crag_retrieval_mode is not None:
            self.retrieval.mode = self.crag_retrieval_mode
        if self.crag_top_k is not None:
            self.retrieval.top_k = self.crag_top_k
        if self.crag_candidate_k is not None:
            self.retrieval.candidate_k = self.crag_candidate_k
        if self.crag_api_host is not None:
            self.api.host = self.crag_api_host
        if self.crag_api_port is not None:
            self.api.port = self.crag_api_port
        if self.crag_context_max_chars is not None:
            self.context.max_chars = self.crag_context_max_chars
        if self.crag_api_key is not None:
            self.api.api_key = self.crag_api_key
        if self.crag_dense_search is not None:
            self.storage.dense_search = self.crag_dense_search
        if self.crag_allowed_index_roots is not None:
            self.api.allowed_index_roots = [
                item.strip()
                for item in self.crag_allowed_index_roots.split(os.pathsep)
                if item.strip()
            ]


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def load_settings(config_path: Path | str | None = None) -> CRAGSettings:
    resolved = Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH
    yaml_data = _load_yaml_config(resolved)
    kwargs: dict[str, Any] = {}
    embedding_data = yaml_data.get("embedding")
    if isinstance(embedding_data, dict):
        kwargs["embedding"] = EmbeddingSettings.model_validate(embedding_data)
    storage_data = yaml_data.get("storage")
    if isinstance(storage_data, dict):
        kwargs["storage"] = StorageSettings.model_validate(storage_data)
    retrieval_data = yaml_data.get("retrieval")
    if isinstance(retrieval_data, dict):
        kwargs["retrieval"] = RetrievalSettings.model_validate(retrieval_data)
    api_data = yaml_data.get("api")
    if isinstance(api_data, dict):
        kwargs["api"] = ApiSettings.model_validate(api_data)
    context_data = yaml_data.get("context")
    if isinstance(context_data, dict):
        kwargs["context"] = ContextSettings.model_validate(context_data)
    return CRAGSettings(**kwargs)


@lru_cache(maxsize=1)
def get_settings() -> CRAGSettings:
    return load_settings()
