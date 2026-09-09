"""Eval report formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def mode_metrics_dict(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "hit_at_k": float(payload["hit_at_k"]),
        "recall_at_k": float(payload["recall_at_k"]),
        "mrr": float(payload["mrr"]),
        "ndcg_at_k": float(payload.get("ndcg_at_k", 0.0)),
    }


def build_comparison(modes: dict[str, Any]) -> dict[str, float]:
    hybrid = modes.get("hybrid", {})
    bm25 = modes.get("bm25", {})
    vector = modes.get("vector", {})
    return {
        "hybrid_minus_bm25_hit": float(hybrid.get("hit_at_k", 0) - bm25.get("hit_at_k", 0)),
        "hybrid_minus_vector_hit": float(hybrid.get("hit_at_k", 0) - vector.get("hit_at_k", 0)),
        "hybrid_minus_bm25_ndcg": float(hybrid.get("ndcg_at_k", 0) - bm25.get("ndcg_at_k", 0)),
        "hybrid_minus_vector_ndcg": float(hybrid.get("ndcg_at_k", 0) - vector.get("ndcg_at_k", 0)),
    }


def _interval(payload: dict[str, Any], metric: str) -> str:
    intervals = payload.get("confidence_intervals") or {}
    bounds = intervals.get(metric)
    if not bounds:
        return ""
    return f"[{float(bounds['low']):.3f}, {float(bounds['high']):.3f}]"


def _run_lines(report: dict[str, Any]) -> list[str]:
    run = report.get("run") or {}
    if not run:
        return []
    environment = run.get("environment") or {}
    retrieval = run.get("retrieval") or {}
    packages = environment.get("packages") or {}
    lines = [
        "## Run",
        "",
        f"- commit: `{environment.get('git_commit', 'unknown')}`",
        f"- crag: `{environment.get('crag_version', '?')}`"
        f" | python: `{environment.get('python', '?')}`",
        f"- platform: `{environment.get('platform', '?')}`",
        f"- embedding model: `{retrieval.get('embedding_model', '?')}`"
        f" via `{retrieval.get('provider', '?')}`",
        f"- dense search: `{retrieval.get('dense_search', '?')}`"
        f" | candidate_k: `{retrieval.get('candidate_k', '?')}`"
        f" | rrf_k: `{retrieval.get('rrf_k', '?')}`",
        f"- chunks retrieved per requested document: `{retrieval.get('chunk_multiplier', '?')}`",
        f"- sentence-transformers: `{packages.get('sentence-transformers', '?')}`"
        f" | torch: `{packages.get('torch', '?')}`",
        f"- index time: `{run.get('index_seconds', '?')}s`"
        f" | total: `{run.get('total_seconds', '?')}s`",
        f"- started: `{run.get('started_at', '?')}`",
    ]
    dataset = report.get("dataset") or {}
    if dataset:
        subset = " (subset)" if dataset.get("subset") else " (full)"
        lines.append(
            f"- dataset: `{dataset.get('dataset_id', '?')}`{subset}"
            f" fingerprint `{dataset.get('fingerprint', '?')}`"
        )
    lines.append("")
    return lines


def report_to_markdown(report: dict[str, Any]) -> str:
    top_k = report.get("top_k")
    lines = [
        f"# CRAG eval report - `{report.get('suite', 'unknown')}`",
        "",
        f"- provider: `{report.get('provider')}`",
        f"- top_k: `{top_k}` documents",
        f"- docs: `{report.get('doc_count', '?')}`",
        f"- queries: `{report.get('query_count', '?')}`",
        f"- rerank: `{report.get('rerank', False)}`",
        "",
        "Metrics are per document: chunks are collapsed to their source document",
        "before the list is cut to top_k. Brackets are 95% bootstrap intervals",
        "over queries.",
        "",
        "| mode | Hit@k | nDCG@k | Recall@k | MRR | p50 ms | p95 ms |",
        "|------|------:|-------:|---------:|----:|-------:|-------:|",
    ]
    for mode, payload in report.get("modes", {}).items():
        latency = payload.get("latency_ms") or {}
        lines.append(
            f"| {mode} "
            f"| {payload['hit_at_k']:.3f} {_interval(payload, 'hit_at_k')} "
            f"| {payload.get('ndcg_at_k', 0.0):.3f} {_interval(payload, 'ndcg_at_k')} "
            f"| {payload['recall_at_k']:.3f} "
            f"| {payload['mrr']:.3f} "
            f"| {latency.get('p50_ms', 0.0):.1f} "
            f"| {latency.get('p95_ms', 0.0):.1f} |"
        )
    comparison = report.get("comparison") or {}
    if comparison:
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- hybrid - bm25 Hit@k: `{comparison.get('hybrid_minus_bm25_hit', 0):+.3f}`",
                f"- hybrid - vector Hit@k: `{comparison.get('hybrid_minus_vector_hit', 0):+.3f}`",
                f"- hybrid - bm25 nDCG@k: `{comparison.get('hybrid_minus_bm25_ndcg', 0):+.3f}`",
                f"- hybrid - vector nDCG@k: `{comparison.get('hybrid_minus_vector_ndcg', 0):+.3f}`",
                "",
            ]
        )
    lines.extend(_run_lines(report))
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
