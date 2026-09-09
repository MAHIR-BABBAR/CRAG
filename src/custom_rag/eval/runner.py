"""Golden-set and benchmark runners for retrieval modes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from custom_rag.core.config import (
    CRAGSettings,
    EmbeddingSettings,
    RetrievalModeName,
    RetrievalSettings,
    StorageSettings,
)
from custom_rag.core.providers.base import EmbeddingProvider
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.core.providers.registry import resolve_embedding_provider
from custom_rag.eval.metrics import (
    MetricResult,
    QueryMetrics,
    evaluate_ranking,
    summarize_query_metrics,
)
from custom_rag.eval.provenance import build_environment
from custom_rag.eval.report import build_comparison, write_report
from custom_rag.ingestion import index_file
from custom_rag.retrieval import retrieve_with_parents
from custom_rag.storage.base import VectorHit
from custom_rag.storage.sqlite_store import SQLiteIndexStore

# Retrieval returns chunks, but relevance is judged per document. Pulling this
# many chunks per requested document leaves room for several chunks of the same
# document to collapse into one ranked result without starving the list.
DEFAULT_CHUNK_MULTIPLIER = 5


@dataclass(frozen=True)
class RelevantRule:
    source_contains: str | None = None
    text_contains: str | None = None
    external_id: str | None = None


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    relevant: list[RelevantRule]
    expect_empty: bool = False
    focus: str | None = None


@dataclass(frozen=True)
class QueryEvalRow:
    query_id: str
    query: str
    mode: RetrievalModeName
    first_relevant_rank: int | None
    relevant_in_top_k: int
    gold_count: int
    hit_ids: list[str]
    latency_ms: float = 0.0


@dataclass(frozen=True)
class ModeEvalReport:
    mode: RetrievalModeName
    metrics: MetricResult
    rows: list[QueryEvalRow]
    latency_ms: dict[str, float] = field(default_factory=dict)


def load_qrels(path: Path | str) -> list[EvalQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = data.get("queries", data)
    if not isinstance(queries, list):
        raise ValueError("qrels must contain a list under 'queries'")
    out: list[EvalQuery] = []
    for item in queries:
        if not isinstance(item, dict):
            raise ValueError("each qrel entry must be an object")
        rules_raw = item.get("relevant", [])
        rules: list[RelevantRule] = []
        for rule in rules_raw:
            rules.append(
                RelevantRule(
                    source_contains=rule.get("source_contains"),
                    text_contains=rule.get("text_contains"),
                    external_id=(
                        None if rule.get("external_id") is None else str(rule["external_id"])
                    ),
                )
            )
        out.append(
            EvalQuery(
                id=str(item["id"]),
                query=str(item["query"]),
                relevant=rules,
                expect_empty=bool(item.get("expect_empty", False)),
                focus=item.get("focus"),
            )
        )
    return out


def hit_is_relevant(
    hit: VectorHit,
    rules: list[RelevantRule],
    *,
    source_path: str,
    external_id: str | None,
    document_text: str | None = None,
) -> bool:
    """Judge whether a retrieved hit's document satisfies the qrel rules.

    ``document_text`` makes ``text_contains`` a statement about the document, so
    a document is not judged irrelevant merely because the ranker surfaced one
    of its other chunks.
    """
    if not rules:
        return False
    if document_text is not None:
        blob = document_text
    else:
        parent_text = hit.parent.text if hit.parent is not None else ""
        blob = f"{hit.chunk.text}\n{parent_text}"
    for rule in rules:
        if rule.external_id is not None:
            if external_id is not None and rule.external_id == external_id:
                return True
            continue
        source_ok = True
        text_ok = True
        if rule.source_contains:
            source_ok = rule.source_contains.lower() in source_path.lower()
        if rule.text_contains:
            text_ok = rule.text_contains.lower() in blob.lower()
        if source_ok and text_ok:
            return True
    return False


def index_corpus(
    paths: list[Path],
    *,
    store: SQLiteIndexStore,
    provider: EmbeddingProvider,
    collection: str = "eval",
) -> None:
    for path in paths:
        extra: dict[str, object] | None = None
        meta_path = path.with_suffix(".meta.json")
        # docs are .txt; sidecar is {stem}.meta.json next to {stem}.txt
        if path.suffix == ".txt":
            meta_path = path.parent / f"{path.stem}.meta.json"
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("external_id") is not None:
                    extra = {"external_id": str(payload["external_id"])}
            except json.JSONDecodeError:
                extra = None
        index_file(
            path,
            provider=provider,
            store=store,
            collection=collection,
            extra_metadata=extra,
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _collapse_to_documents(
    hits: list[VectorHit],
    *,
    store: SQLiteIndexStore,
    rules: list[RelevantRule],
    limit: int,
) -> tuple[list[float], list[str]]:
    """Reduce a ranked list of chunks to a ranked list of documents.

    Several chunks of one document are one retrieved document, not several. Left
    uncollapsed, a document split into ten chunks would fill the whole result
    list and inflate recall and nDCG well past what the ranking earned.

    A document ranks where its best chunk ranked, and counts as relevant if any
    of its retrieved chunks satisfies the rules: relevance belongs to the
    document, not to whichever of its chunks happened to surface first.
    """
    needs_text = any(rule.text_contains for rule in rules)
    ordered_keys: list[str] = []
    relevant_keys: set[str] = set()
    seen: set[str] = set()
    for hit in hits:
        external_id = store.get_external_id(hit.doc_id)
        key = external_id or hit.doc_id
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)
        if key in relevant_keys:
            continue
        source = store.get_source_path(hit.doc_id) or ""
        if hit_is_relevant(
            hit,
            rules,
            source_path=source,
            external_id=external_id,
            document_text=store.get_document_text(hit.doc_id) if needs_text else None,
        ):
            relevant_keys.add(key)

    doc_keys = ordered_keys[:limit]
    relevance = [1.0 if key in relevant_keys else 0.0 for key in doc_keys]
    return relevance, doc_keys


def evaluate_mode(
    queries: list[EvalQuery],
    *,
    store: SQLiteIndexStore,
    mode: RetrievalModeName,
    provider: EmbeddingProvider | None,
    settings: CRAGSettings,
    collection: str = "eval",
    top_k: int = 5,
    rerank: bool = False,
    chunk_multiplier: int = DEFAULT_CHUNK_MULTIPLIER,
) -> ModeEvalReport:
    """Evaluate one retrieval mode at document level.

    Both the reranked and non-reranked arms retrieve the same candidate pool, so
    a difference between them is the cross-encoder's ordering rather than one arm
    having been shown more documents than the other.
    """
    pool_k = max(top_k * chunk_multiplier, top_k)
    run_settings = settings.model_copy(deep=True)
    run_settings.retrieval.rerank_top_n = pool_k
    candidate_k = max(run_settings.retrieval.candidate_k, pool_k)

    per_query: list[QueryMetrics] = []
    rows: list[QueryEvalRow] = []
    latencies: list[float] = []

    for item in queries:
        started = time.perf_counter()
        result = retrieve_with_parents(
            item.query,
            store=store,
            provider=provider,
            top_k=pool_k,
            candidate_k=candidate_k,
            collection=collection,
            mode=mode,
            settings=run_settings,
            rerank=rerank,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        relevance, doc_keys = _collapse_to_documents(
            result.hits, store=store, rules=item.relevant, limit=top_k
        )
        gold = 0 if item.expect_empty else max(1, len(item.relevant))
        per_query.append(evaluate_ranking(relevance, gold, k=top_k))

        found = sum(1 for value in relevance if value > 0.0)
        first_rank = next(
            (index for index, value in enumerate(relevance, start=1) if value > 0.0),
            None,
        )
        if item.expect_empty:
            first_rank = 1 if found == 0 else None
        rows.append(
            QueryEvalRow(
                query_id=item.id,
                query=item.query,
                mode=mode,
                first_relevant_rank=first_rank,
                relevant_in_top_k=found,
                gold_count=gold,
                hit_ids=doc_keys,
                latency_ms=round(latency_ms, 2),
            )
        )

    metrics = summarize_query_metrics(per_query, k=top_k)
    latency_stats = {
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
    }
    return ModeEvalReport(mode=mode, metrics=metrics, rows=rows, latency_ms=latency_stats)


def default_fixture_paths(fixtures_dir: Path) -> list[Path]:
    names = ["sample.md", "sample.txt", "sample.py", "sample.html"]
    return [fixtures_dir / name for name in names if (fixtures_dir / name).is_file()]


def _mode_payload(report: ModeEvalReport) -> dict[str, Any]:
    return {
        "hit_at_k": report.metrics.hit_at_k,
        "recall_at_k": report.metrics.recall_at_k,
        "mrr": report.metrics.mrr,
        "ndcg_at_k": report.metrics.ndcg_at_k,
        "queries": report.metrics.queries,
        "confidence_intervals": {
            name: interval.as_dict() for name, interval in report.metrics.intervals.items()
        },
        "latency_ms": report.latency_ms,
        "rows": [
            {
                "query_id": row.query_id,
                "first_relevant_rank": row.first_relevant_rank,
                "relevant_in_top_k": row.relevant_in_top_k,
                "gold_count": row.gold_count,
                "latency_ms": row.latency_ms,
            }
            for row in report.rows
        ],
    }


def _resolve_provider(settings: CRAGSettings, provider_name: str) -> EmbeddingProvider:
    if provider_name == "mock":
        return MockEmbeddingProvider()
    return resolve_embedding_provider(settings, provider=provider_name)


def _run_metadata(
    *,
    settings: CRAGSettings,
    provider_name: str,
    top_k: int,
    chunk_multiplier: int,
    rerank: bool,
) -> dict[str, Any]:
    return {
        "environment": build_environment(),
        "started_at": datetime.now(tz=UTC).isoformat(),
        "retrieval": {
            "provider": provider_name,
            "embedding_model": settings.embedding.model,
            "top_k": top_k,
            "candidate_k": settings.retrieval.candidate_k,
            "rrf_k": settings.retrieval.rrf_k,
            "chunk_multiplier": chunk_multiplier,
            "dense_search": settings.storage.dense_search,
            "rerank": rerank,
            "rerank_model": settings.retrieval.rerank_model if rerank else None,
        },
        "grading": {
            "unit": "document",
            "note": (
                "Chunks are collapsed to their source document before cutting to "
                "top_k, so metrics are per document and comparable to document-level "
                "IR results."
            ),
        },
    }


def run_eval_suite(
    *,
    qrels_path: Path,
    fixtures_dir: Path,
    db_path: Path,
    settings: CRAGSettings | None = None,
    provider_name: str = "mock",
    modes: list[RetrievalModeName] | None = None,
    top_k: int = 5,
    collection: str = "eval",
    chunk_multiplier: int = DEFAULT_CHUNK_MULTIPLIER,
) -> dict[str, Any]:
    active_settings = settings or CRAGSettings(
        embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
        storage=StorageSettings(path=str(db_path), collection=collection),
    )
    provider = _resolve_provider(active_settings, provider_name)
    started = time.perf_counter()
    with SQLiteIndexStore(
        db_path,
        default_collection=collection,
        dense_search=active_settings.storage.dense_search,
    ) as store:
        paths = default_fixture_paths(fixtures_dir)
        index_corpus(paths, store=store, provider=provider, collection=collection)
        index_seconds = time.perf_counter() - started
        queries = load_qrels(qrels_path)
        active_modes: list[RetrievalModeName] = modes or ["vector", "bm25", "hybrid"]
        reports: dict[str, Any] = {
            "suite": "fixtures",
            "provider": provider_name,
            "top_k": top_k,
            "doc_count": len(paths),
            "query_count": len(queries),
            "rerank": False,
            "run": _run_metadata(
                settings=active_settings,
                provider_name=provider_name,
                top_k=top_k,
                chunk_multiplier=chunk_multiplier,
                rerank=False,
            ),
            "modes": {},
        }
        for mode in active_modes:
            mode_provider = None if mode == "bm25" else provider
            report = evaluate_mode(
                queries,
                store=store,
                mode=mode,
                provider=mode_provider,
                settings=active_settings,
                collection=collection,
                top_k=top_k,
                chunk_multiplier=chunk_multiplier,
            )
            reports["modes"][mode] = _mode_payload(report)
        reports["comparison"] = build_comparison(reports["modes"])
        reports["run"]["index_seconds"] = round(index_seconds, 2)
        reports["run"]["total_seconds"] = round(time.perf_counter() - started, 2)
        return reports


def run_benchmark_eval(
    *,
    suite: str = "scifact",
    db_path: Path,
    provider_name: str = "mock",
    top_k: int = 10,
    cache_dir: Path | None = None,
    modes: list[RetrievalModeName] | None = None,
    collection: str = "benchmark",
    force_download: bool = False,
    max_docs: int | None = None,
    max_queries: int | None = None,
    report_path: Path | None = None,
    rerank: bool = False,
    chunk_multiplier: int = DEFAULT_CHUNK_MULTIPLIER,
) -> dict[str, Any]:
    """Prepare a public benchmark corpus, index it, evaluate modes, write a report."""
    if suite != "scifact":
        raise ValueError(f"unsupported suite: {suite!r} (supported: scifact)")

    from custom_rag.eval.datasets import prepare_scifact

    prepared = prepare_scifact(
        cache_dir,
        force=force_download,
        max_docs=max_docs,
        max_queries=max_queries,
    )
    active_settings = CRAGSettings(
        embedding=EmbeddingSettings(
            provider="mock" if provider_name == "mock" else provider_name,  # type: ignore[arg-type]
            model="mock-embed-v1" if provider_name == "mock" else "all-MiniLM-L6-v2",
        ),
        storage=StorageSettings(path=str(db_path), collection=collection),
        retrieval=RetrievalSettings(rerank_enabled=rerank),
    )
    provider = _resolve_provider(active_settings, provider_name)
    started = time.perf_counter()
    with SQLiteIndexStore(
        db_path,
        default_collection=collection,
        dense_search=active_settings.storage.dense_search,
    ) as store:
        doc_paths = sorted(prepared.docs_dir.glob("*.txt"))
        index_corpus(doc_paths, store=store, provider=provider, collection=collection)
        index_seconds = time.perf_counter() - started
        queries = load_qrels(prepared.qrels_path)
        active_modes: list[RetrievalModeName] = modes or ["vector", "bm25", "hybrid"]
        report: dict[str, Any] = {
            "suite": prepared.name,
            "provider": provider_name,
            "top_k": top_k,
            "doc_count": len(doc_paths),
            "query_count": len(queries),
            "cache_dir": str(prepared.cache_dir),
            "rerank": rerank,
            "dataset": prepared.describe(),
            "run": _run_metadata(
                settings=active_settings,
                provider_name=provider_name,
                top_k=top_k,
                chunk_multiplier=chunk_multiplier,
                rerank=rerank,
            ),
            "modes": {},
        }
        for mode in active_modes:
            mode_provider = None if mode == "bm25" else provider
            mode_report = evaluate_mode(
                queries,
                store=store,
                mode=mode,
                provider=mode_provider,
                settings=active_settings,
                collection=collection,
                top_k=top_k,
                rerank=rerank,
                chunk_multiplier=chunk_multiplier,
            )
            report["modes"][mode] = _mode_payload(mode_report)
        report["comparison"] = build_comparison(report["modes"])
        report["run"]["index_seconds"] = round(index_seconds, 2)
        report["run"]["total_seconds"] = round(time.perf_counter() - started, 2)
        if report_path is not None:
            write_report(report, report_path)
        return report
