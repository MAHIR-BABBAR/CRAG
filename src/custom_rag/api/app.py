"""FastAPI application factory for the CRAG open socket."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from custom_rag import __version__
from custom_rag.api.auth import require_api_key
from custom_rag.api.deps import (
    ProviderCache,
    get_provider_cache,
    get_settings,
    get_store,
    resolve_index_path,
)
from custom_rag.api.schemas import (
    CitationModel,
    ContextRequest,
    ContextResponse,
    HealthResponse,
    IndexBatchResponse,
    IndexRequest,
    IndexResponse,
    QualityStatsModel,
    RetrieveRequest,
    RetrieveResponse,
)
from custom_rag.core.config import CRAGSettings, RetrievalModeName, load_settings
from custom_rag.core.exceptions import ConfigError, CRAGError, StorageError
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.ingestion import EmbedConfig, index_path
from custom_rag.retrieval import retrieve_with_parents
from custom_rag.retrieval.packer import assemble_context
from custom_rag.retrieval.quality import score_retrieval
from custom_rag.storage.sqlite_store import SQLiteIndexStore

logger = logging.getLogger("custom_rag.api")


def _internal_error(operation: str, exc: Exception) -> HTTPException:
    """Log the cause, return an opaque 500.

    Exception text here can contain absolute paths and connection strings, so it
    goes to the server log rather than to the client.
    """
    logger.exception("%s failed", operation, exc_info=exc)
    return HTTPException(status_code=500, detail=f"{operation} failed")


def create_app(settings: CRAGSettings | None = None) -> FastAPI:
    """Build the FastAPI app, optionally with injected settings (tests)."""
    active_settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = SQLiteIndexStore(
            active_settings.storage.path,
            default_collection=active_settings.storage.collection,
            dense_search=active_settings.storage.dense_search,
        )
        app.state.settings = active_settings
        app.state.store = store
        app.state.providers = ProviderCache(active_settings)
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="CRAG Open Socket", version=__version__, lifespan=lifespan)

    @app.get("/v1/health", response_model=HealthResponse)
    def health(
        request: Request,
        cfg: CRAGSettings = Depends(get_settings),
        store: SQLiteIndexStore = Depends(get_store),
    ) -> HealthResponse | JSONResponse:
        _ = request
        try:
            store.ping()
        except StorageError:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "storage_path": cfg.storage.path,
                    "collection": cfg.storage.collection,
                },
            )
        return HealthResponse(
            status="ok",
            version=__version__,
            storage_path=cfg.storage.path,
            collection=cfg.storage.collection,
        )

    @app.post(
        "/v1/index",
        response_model=IndexResponse | IndexBatchResponse,
        dependencies=[Depends(require_api_key)],
    )
    def index_document(
        body: IndexRequest,
        cfg: CRAGSettings = Depends(get_settings),
        store: SQLiteIndexStore = Depends(get_store),
        providers: ProviderCache = Depends(get_provider_cache),
    ) -> IndexResponse | IndexBatchResponse:
        try:
            path = resolve_index_path(body.path, cfg)
        except ConfigError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"path not found: {body.path}")

        collection = body.collection or cfg.storage.collection
        try:
            provider = providers.get(body.provider)
            embed_config = EmbedConfig(provider=body.provider) if body.provider else None
            result = index_path(
                path,
                embed_config=embed_config,
                provider=provider,
                store=store,
                collection=collection,
                recursive=body.recursive,
                max_files=cfg.api.max_index_files,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CRAGError as exc:
            raise _internal_error("index", exc) from exc
        except Exception as exc:
            raise _internal_error("index", exc) from exc

        if path.is_file():
            if result.failed:
                raise HTTPException(status_code=422, detail=result.failed[0]["error"])
            if not result.files:
                raise HTTPException(status_code=415, detail="unsupported or empty file")
            first = result.files[0]
            return IndexResponse(
                skipped=first.skipped,
                doc_id=first.doc_id or "",
                source_path=first.path,
                children=first.children,
                parents=first.parents,
                collection=collection,
            )

        return IndexBatchResponse(
            indexed=result.indexed,
            skipped=result.skipped,
            failed=result.failed,
            file_count=len(result.files),
            truncated=result.truncated,
            max_files=cfg.api.max_index_files,
            collection=collection,
            path=str(path),
        )

    def _validate_depths(cfg: CRAGSettings, top_k: int, candidate_k: int) -> None:
        if top_k > cfg.retrieval.max_top_k:
            raise HTTPException(
                status_code=422,
                detail=f"top_k exceeds max_top_k ({cfg.retrieval.max_top_k})",
            )
        if candidate_k > cfg.retrieval.max_candidate_k:
            raise HTTPException(
                status_code=422,
                detail=(f"candidate_k exceeds max_candidate_k ({cfg.retrieval.max_candidate_k})"),
            )

    def _provider_for_mode(
        mode: RetrievalModeName,
        providers: ProviderCache,
        requested: RetrieveRequest | ContextRequest,
    ) -> EmbeddingProvider | None:
        if mode == "bm25":
            return None
        try:
            return providers.get(requested.provider)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/retrieve",
        response_model=RetrieveResponse,
        dependencies=[Depends(require_api_key)],
    )
    def retrieve(
        body: RetrieveRequest,
        cfg: CRAGSettings = Depends(get_settings),
        store: SQLiteIndexStore = Depends(get_store),
        providers: ProviderCache = Depends(get_provider_cache),
    ) -> RetrieveResponse:
        mode: RetrievalModeName = body.mode or cfg.retrieval.mode
        top_k = body.top_k or cfg.retrieval.top_k
        candidate_k = body.candidate_k or cfg.retrieval.candidate_k
        collection = body.collection or cfg.storage.collection
        _validate_depths(cfg, top_k, candidate_k)
        provider = _provider_for_mode(mode, providers, body)

        try:
            result = retrieve_with_parents(
                body.query,
                store=store,
                provider=provider,
                top_k=top_k,
                candidate_k=candidate_k,
                collection=collection,
                mode=mode,
                settings=cfg,
                rerank=body.rerank,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CRAGError as exc:
            raise _internal_error("retrieve", exc) from exc
        except Exception as exc:
            raise _internal_error("retrieve", exc) from exc

        return RetrieveResponse(
            query=result.query,
            mode=result.mode,
            hits=result.as_dicts(),
            reranked=result.reranked,
        )

    @app.post(
        "/v1/context",
        response_model=ContextResponse,
        dependencies=[Depends(require_api_key)],
    )
    def context(
        body: ContextRequest,
        cfg: CRAGSettings = Depends(get_settings),
        store: SQLiteIndexStore = Depends(get_store),
        providers: ProviderCache = Depends(get_provider_cache),
    ) -> ContextResponse:
        mode: RetrievalModeName = body.mode or cfg.retrieval.mode
        top_k = body.top_k or cfg.retrieval.top_k
        candidate_k = body.candidate_k or cfg.retrieval.candidate_k
        collection = body.collection or cfg.storage.collection
        max_chars = body.max_chars or cfg.context.max_chars
        _validate_depths(cfg, top_k, candidate_k)
        provider = _provider_for_mode(mode, providers, body)

        try:
            result = retrieve_with_parents(
                body.query,
                store=store,
                provider=provider,
                top_k=top_k,
                candidate_k=candidate_k,
                collection=collection,
                mode=mode,
                settings=cfg,
                rerank=body.rerank,
            )
            packed = assemble_context(
                result.hits,
                max_chars=max_chars,
                include_parents=cfg.context.include_parents,
                source_path_for=store.get_source_path,
            )
            quality = score_retrieval(
                result.hits,
                packed,
                mode=mode,
                min_score_high=cfg.context.min_score_high,
                min_bm25_high=cfg.context.min_bm25_high,
                min_rrf_high=cfg.context.min_rrf_high,
                reranked=result.reranked,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CRAGError as exc:
            raise _internal_error("context", exc) from exc
        except Exception as exc:
            raise _internal_error("context", exc) from exc

        return ContextResponse(
            query=result.query,
            mode=result.mode,
            context=packed.text,
            citations=[
                CitationModel(
                    index=c.index,
                    doc_id=c.doc_id,
                    chunk_id=c.chunk_id,
                    parent_chunk_id=c.parent_chunk_id,
                    source_path=c.source_path,
                    score=c.score,
                    page=c.page,
                    slide=c.slide,
                )
                for c in packed.citations
            ],
            retrieval_quality=quality.label,
            quality_stats=QualityStatsModel(
                hit_count=quality.stats.hit_count,
                chars_used=quality.stats.chars_used,
                truncated=quality.stats.truncated,
                top_score=quality.stats.top_score,
            ),
            hits=result.as_dicts(),
        )

    return app


# Module-level app for `uvicorn custom_rag.api.app:app`
app: FastAPI = create_app()
