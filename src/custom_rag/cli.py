"""CRAG command-line interface."""

from __future__ import annotations

import ipaddress
import json
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import click

from custom_rag import __version__

if TYPE_CHECKING:
    from custom_rag.core.config import CRAGSettings
    from custom_rag.core.providers.base import EmbeddingProvider


def _provider_for(settings: CRAGSettings, provider: str | None) -> EmbeddingProvider:
    """Resolve a provider against the loaded settings, not a fresh default set."""
    from custom_rag.core.providers.registry import resolve_embedding_provider

    return resolve_embedding_provider(settings, provider=provider)


@click.group()
@click.version_option(version=__version__, prog_name="crag")
def main() -> None:
    """Index documents, search them, and serve retrieval over HTTP."""


@main.command("parse")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit ParsedDocument as JSON.")
def parse(path: Path, as_json: bool) -> None:
    """Parse a document and print block summary or JSON."""
    from custom_rag.ingestion import parse_file

    document = parse_file(path)
    if as_json:
        click.echo(json.dumps(document.model_dump(mode="json"), indent=2))
        return

    click.echo(f"doc_type: {document.metadata.doc_type}")
    click.echo(f"blocks: {len(document.blocks)}")
    click.echo(f"content_hash: {document.metadata.content_hash}")
    for block in document.blocks[:10]:
        preview = block.text[:80].replace("\n", " ")
        click.echo(f"  [{block.block_type}] {block.block_id}: {preview}")
    if len(document.blocks) > 10:
        click.echo(f"  ... and {len(document.blocks) - 10} more blocks")


@main.command("chunk")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit ChunkedDocument as JSON.")
def chunk(path: Path, as_json: bool) -> None:
    """Parse and chunk a document; print chunk summary or JSON."""
    from custom_rag.ingestion import chunk_file

    chunked = chunk_file(path)
    if as_json:
        click.echo(json.dumps(chunked.model_dump(mode="json"), indent=2))
        return

    click.echo(f"doc_type: {chunked.metadata.doc_type}")
    click.echo(f"chunks: {len(chunked.chunks)}")
    click.echo(f"parents: {len(chunked.parents())}")
    click.echo(f"children: {len(chunked.children())}")
    for item in chunked.chunks[:10]:
        preview = item.text[:80].replace("\n", " ")
        parent = item.parent_chunk_id or "-"
        click.echo(f"  [{item.role}] {item.chunk_id} (parent={parent}): {preview}")
    if len(chunked.chunks) > 10:
        click.echo(f"  ... and {len(chunked.chunks) - 10} more chunks")


@main.command("embed")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit EmbeddedDocument as JSON.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "local", "mock"], case_sensitive=False),
    default=None,
    help="Embedding provider override.",
)
def embed(path: Path, as_json: bool, provider: str | None) -> None:
    """Parse, chunk, and embed a document; print summary or JSON."""
    from custom_rag.core.config import get_settings
    from custom_rag.ingestion import EmbedConfig, embed_file

    settings = get_settings()
    embed_config = EmbedConfig(provider=provider) if provider else None
    embedded = embed_file(
        path,
        embed_config=embed_config,
        provider=_provider_for(settings, provider),
    )

    if as_json:
        click.echo(json.dumps(embedded.model_dump(mode="json"), indent=2))
        return

    click.echo(f"doc_type: {embedded.metadata.doc_type}")
    click.echo(f"skipped: {embedded.skipped}")
    click.echo(f"embedded_children: {len(embedded.children)}")
    click.echo(f"parents: {len(embedded.parents)}")
    if embedded.children:
        sample = embedded.children[0]
        preview = sample.embedding[:5]
        click.echo(f"dimensions: {sample.dimensions}")
        click.echo(f"provider: {sample.provider}")
        click.echo(f"model: {sample.model}")
        click.echo(f"sample vector prefix: {preview}")


@main.command("index")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--provider",
    type=click.Choice(["openai", "local", "mock"], case_sensitive=False),
    default=None,
    help="Embedding provider override.",
)
@click.option("--collection", default=None, help="Index collection name.")
@click.option(
    "--recursive/--no-recursive",
    default=True,
    show_default=True,
    help="When path is a directory, walk recursively.",
)
def index(path: Path, provider: str | None, collection: str | None, recursive: bool) -> None:
    """Parse, chunk, embed, and persist a file or directory into the local index."""
    from custom_rag.core.config import get_settings
    from custom_rag.ingestion import EmbedConfig, index_path
    from custom_rag.storage import SQLiteIndexStore

    settings = get_settings()
    embed_config = EmbedConfig(provider=provider) if provider else None
    with closing(
        SQLiteIndexStore(
            settings.storage.path,
            default_collection=collection or settings.storage.collection,
            dense_search=settings.storage.dense_search,
        )
    ) as store:
        result = index_path(
            path,
            embed_config=embed_config,
            provider=_provider_for(settings, provider),
            store=store,
            collection=collection,
            recursive=recursive,
            max_files=settings.api.max_index_files,
        )
    click.echo(f"indexed: {result.indexed}")
    click.echo(f"skipped: {result.skipped}")
    click.echo(f"failed: {len(result.failed)}")
    click.echo(f"collection: {collection or settings.storage.collection}")
    click.echo(f"db: {settings.storage.path}")
    if result.truncated:
        click.echo(
            f"warning: stopped at api.max_index_files ({settings.api.max_index_files}); "
            "the directory holds more indexable files"
        )
    for item in result.failed[:10]:
        click.echo(f"  fail {item['path']}: {item['error']}")
    if len(result.failed) > 10:
        click.echo(f"  ... and {len(result.failed) - 10} more failures")


@main.command("query")
@click.argument("query_text")
@click.option("--top-k", default=None, type=int, help="Number of child hits to return.")
@click.option(
    "--candidate-k",
    default=None,
    type=int,
    help="Per-channel depth before RRF (hybrid mode).",
)
@click.option(
    "--mode",
    type=click.Choice(["vector", "bm25", "hybrid"], case_sensitive=False),
    default=None,
    help="Retrieval mode (default: config retrieval.mode).",
)
@click.option("--collection", default=None, help="Index collection name.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "local", "mock"], case_sensitive=False),
    default=None,
    help="Embedding provider for the query vector.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit retrieval results as JSON.")
@click.option("--rerank/--no-rerank", default=None, help="Cross-encoder rerank (default: config).")
def query(
    query_text: str,
    top_k: int | None,
    candidate_k: int | None,
    mode: str | None,
    collection: str | None,
    provider: str | None,
    as_json: bool,
    rerank: bool | None,
) -> None:
    """Search the index and return matched children with owning parents."""
    from custom_rag.core.config import get_settings
    from custom_rag.retrieval import retrieve_with_parents
    from custom_rag.retrieval.context import format_hit_preview
    from custom_rag.storage import SQLiteIndexStore

    settings = get_settings()
    active_mode = mode.lower() if mode else settings.retrieval.mode
    active_provider = None if active_mode == "bm25" else _provider_for(settings, provider)
    with closing(
        SQLiteIndexStore(
            settings.storage.path,
            default_collection=collection or settings.storage.collection,
            dense_search=settings.storage.dense_search,
        )
    ) as store:
        result = retrieve_with_parents(
            query_text,
            store=store,
            provider=active_provider,
            top_k=top_k,
            candidate_k=candidate_k,
            collection=collection,
            mode=active_mode,  # type: ignore[arg-type]
            settings=settings,
            rerank=rerank,
        )
    if as_json:
        click.echo(
            json.dumps(
                {"query": result.query, "mode": result.mode, "hits": result.as_dicts()},
                indent=2,
            )
        )
        return

    click.echo(f"query: {result.query}")
    click.echo(f"mode: {result.mode}")
    click.echo(f"hits: {len(result.hits)}")
    for index, hit in enumerate(result.hits, start=1):
        child_preview, parent_preview = format_hit_preview(hit)
        click.echo(f"  {index}. score={hit.score:.4f} child={hit.chunk_id}")
        click.echo(f"     child: {child_preview}")
        parent_id = hit.parent.chunk_id if hit.parent is not None else "-"
        click.echo(f"     parent={parent_id}: {parent_preview}")


@main.command("eval")
@click.option(
    "--suite",
    type=click.Choice(["fixtures", "scifact"], case_sensitive=False),
    default="fixtures",
    show_default=True,
    help="fixtures = offline golden set; scifact = download BEIR SciFact.",
)
@click.option(
    "--qrels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to qrels JSON (fixtures suite default: tests/eval/qrels.json).",
)
@click.option(
    "--fixtures",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Fixtures directory to index (fixtures suite only).",
)
@click.option("--top-k", default=None, type=int, help="Default 5 (fixtures) or 10 (scifact).")
@click.option(
    "--provider",
    type=click.Choice(["mock", "local", "openai"], case_sensitive=False),
    default="mock",
    show_default=True,
)
@click.option(
    "--report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSON + Markdown report to this path.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit full report as JSON.")
@click.option("--force-download", is_flag=True, help="Re-download SciFact cache.")
@click.option("--max-docs", default=None, type=int, help="Limit SciFact docs (dev/smoke).")
@click.option("--max-queries", default=None, type=int, help="Limit SciFact queries (dev/smoke).")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder rerank during eval.")
@click.option(
    "--chunk-multiplier",
    default=None,
    type=int,
    help="Chunks retrieved per requested document before collapsing (default 5).",
)
@click.option(
    "--db-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Reuse an index across runs instead of building a throwaway one.",
)
def eval_cmd(
    suite: str,
    qrels: Path | None,
    fixtures: Path | None,
    top_k: int | None,
    provider: str,
    report: Path | None,
    as_json: bool,
    force_download: bool,
    max_docs: int | None,
    max_queries: int | None,
    rerank: bool,
    chunk_multiplier: int | None,
    db_path: Path | None,
) -> None:
    """Run document-level retrieval eval (Hit@k, Recall@k, MRR, nDCG@k) across modes."""
    import tempfile
    from contextlib import nullcontext

    from custom_rag.eval.report import report_to_markdown, write_report
    from custom_rag.eval.runner import (
        DEFAULT_CHUNK_MULTIPLIER,
        run_benchmark_eval,
        run_eval_suite,
    )

    suite_name = suite.lower()
    active_top_k = top_k if top_k is not None else (10 if suite_name == "scifact" else 5)
    active_multiplier = chunk_multiplier or DEFAULT_CHUNK_MULTIPLIER

    # A reused index lets an ablation vary one setting without paying to embed
    # the corpus again; unchanged files are skipped by content hash.
    scratch = nullcontext(None) if db_path else tempfile.TemporaryDirectory(prefix="crag-eval-")

    with scratch as tmp:
        active_db = db_path or (Path(str(tmp)) / "eval.sqlite3")
        if suite_name == "scifact":
            result = run_benchmark_eval(
                suite="scifact",
                db_path=active_db,
                provider_name=provider.lower(),
                top_k=active_top_k,
                force_download=force_download,
                max_docs=max_docs,
                max_queries=max_queries,
                report_path=report,
                rerank=rerank,
                chunk_multiplier=active_multiplier,
            )
        else:
            repo_root = Path(__file__).resolve().parents[2]
            qrels_path = qrels or (repo_root / "tests" / "eval" / "qrels.json")
            fixtures_dir = fixtures or (repo_root / "tests" / "fixtures")
            if not qrels_path.is_file():
                raise click.ClickException(f"qrels not found: {qrels_path}")
            if not fixtures_dir.is_dir():
                raise click.ClickException(f"fixtures not found: {fixtures_dir}")
            result = run_eval_suite(
                qrels_path=qrels_path,
                fixtures_dir=fixtures_dir,
                db_path=active_db,
                provider_name=provider.lower(),
                top_k=active_top_k,
                chunk_multiplier=active_multiplier,
            )
            if report is not None:
                write_report(result, report)

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(report_to_markdown(result))


@main.command("context")
@click.argument("query_text")
@click.option("--top-k", default=None, type=int, help="Number of child hits to return.")
@click.option(
    "--candidate-k",
    default=None,
    type=int,
    help="Per-channel depth before RRF (hybrid mode).",
)
@click.option(
    "--mode",
    type=click.Choice(["vector", "bm25", "hybrid"], case_sensitive=False),
    default=None,
    help="Retrieval mode (default: config retrieval.mode).",
)
@click.option("--max-chars", default=None, type=int, help="Packed context char budget.")
@click.option("--collection", default=None, help="Index collection name.")
@click.option(
    "--provider",
    type=click.Choice(["openai", "local", "mock"], case_sensitive=False),
    default=None,
    help="Embedding provider for the query vector.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit packed context as JSON.")
@click.option("--rerank/--no-rerank", default=None, help="Cross-encoder rerank (default: config).")
def context_cmd(
    query_text: str,
    top_k: int | None,
    candidate_k: int | None,
    mode: str | None,
    max_chars: int | None,
    collection: str | None,
    provider: str | None,
    as_json: bool,
    rerank: bool | None,
) -> None:
    """Retrieve, pack context under a char budget, and emit citations."""
    from custom_rag.core.config import get_settings
    from custom_rag.retrieval import assemble_context, retrieve_with_parents, score_retrieval
    from custom_rag.storage import SQLiteIndexStore

    settings = get_settings()
    active_mode = mode.lower() if mode else settings.retrieval.mode
    active_provider = None if active_mode == "bm25" else _provider_for(settings, provider)
    with closing(
        SQLiteIndexStore(
            settings.storage.path,
            default_collection=collection or settings.storage.collection,
            dense_search=settings.storage.dense_search,
        )
    ) as store:
        result = retrieve_with_parents(
            query_text,
            store=store,
            provider=active_provider,
            top_k=top_k,
            candidate_k=candidate_k,
            collection=collection,
            mode=active_mode,  # type: ignore[arg-type]
            settings=settings,
            rerank=rerank,
        )
        packed = assemble_context(
            result.hits,
            max_chars=max_chars or settings.context.max_chars,
            include_parents=settings.context.include_parents,
            source_path_for=store.get_source_path,
        )
    quality = score_retrieval(
        result.hits,
        packed,
        mode=result.mode,
        min_score_high=settings.context.min_score_high,
        min_bm25_high=settings.context.min_bm25_high,
        min_rrf_high=settings.context.min_rrf_high,
        reranked=result.reranked,
    )
    if as_json:
        click.echo(
            json.dumps(
                {
                    "query": result.query,
                    "mode": result.mode,
                    "context": packed.text,
                    "citations": [
                        {
                            "index": c.index,
                            "doc_id": c.doc_id,
                            "chunk_id": c.chunk_id,
                            "parent_chunk_id": c.parent_chunk_id,
                            "source_path": c.source_path,
                            "score": c.score,
                            "page": c.page,
                            "slide": c.slide,
                        }
                        for c in packed.citations
                    ],
                    "retrieval_quality": quality.label,
                    "quality_stats": {
                        "hit_count": quality.stats.hit_count,
                        "chars_used": quality.stats.chars_used,
                        "truncated": quality.stats.truncated,
                        "top_score": quality.stats.top_score,
                    },
                    "hits": result.as_dicts(),
                },
                indent=2,
            )
        )
        return

    click.echo(f"query: {result.query}")
    click.echo(f"mode: {result.mode}")
    click.echo(f"retrieval_quality: {quality.label}")
    click.echo(f"chars_used: {quality.stats.chars_used}")
    click.echo("--- context ---")
    click.echo(packed.text or "(empty)")
    if packed.citations:
        click.echo("--- citations ---")
        for cite in packed.citations:
            click.echo(
                f"  [{cite.index}] {cite.source_path} chunk={cite.chunk_id} score={cite.score:.4f}"
            )


@main.command("serve")
@click.option("--host", default=None, help="Bind host (default: config api.host).")
@click.option("--port", default=None, type=int, help="Bind port (default: config api.port).")
def serve(host: str | None, port: int | None) -> None:
    """Run the FastAPI open socket for external agents."""
    import uvicorn

    from custom_rag.api.app import create_app
    from custom_rag.core.config import get_settings

    settings = get_settings()
    bind_host = host or settings.api.host
    bind_port = port or settings.api.port
    if not _is_loopback_host(bind_host) and not settings.api.api_key:
        click.echo(
            "warning: binding outside loopback with no API key set. Any client that "
            "can reach this host can index local files and read the index. Set "
            "CRAG_API_KEY, and api.allowed_index_roots to limit what /v1/index reads.",
            err=True,
        )
    app = create_app(settings)
    click.echo(f"serving CRAG open socket on http://{bind_host}:{bind_port}")
    uvicorn.run(app, host=bind_host, port=bind_port)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


if __name__ == "__main__":
    main()
