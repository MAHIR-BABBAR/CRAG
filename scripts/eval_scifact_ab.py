"""Index SciFact once; evaluate without and with CE rerank."""

from __future__ import annotations

import tempfile
from pathlib import Path

from custom_rag.core.config import (
    CRAGSettings,
    EmbeddingSettings,
    StorageSettings,
)
from custom_rag.core.providers.registry import resolve_embedding_provider
from custom_rag.eval.datasets import prepare_scifact
from custom_rag.eval.report import build_comparison, write_report
from custom_rag.eval.runner import evaluate_mode, index_corpus, load_qrels
from custom_rag.storage.sqlite_store import SQLiteIndexStore


def main() -> None:
    prepared = prepare_scifact()  # reuse cache
    queries = load_qrels(prepared.qrels_path)
    doc_paths = sorted(prepared.docs_dir.glob("*.txt"))
    top_k = 10
    modes = ("vector", "bm25", "hybrid")

    with tempfile.TemporaryDirectory(prefix="crag-scifact-ab-") as tmp:
        db_path = Path(tmp) / "eval.sqlite3"
        settings = CRAGSettings(
            embedding=EmbeddingSettings(provider="local", model="all-MiniLM-L6-v2"),
            storage=StorageSettings(path=str(db_path), collection="benchmark"),
        )
        provider = resolve_embedding_provider(settings, provider="local")
        store = SQLiteIndexStore(db_path, default_collection="benchmark")
        try:
            print(f"indexing {len(doc_paths)} docs...", flush=True)
            index_corpus(doc_paths, store=store, provider=provider, collection="benchmark")
            for rerank, out in (
                (False, Path("reports/scifact_local.json")),
                (True, Path("reports/scifact_local_rerank.json")),
            ):
                print(f"evaluating rerank={rerank}...", flush=True)
                report = {
                    "suite": "scifact",
                    "provider": "local",
                    "top_k": top_k,
                    "doc_count": len(doc_paths),
                    "query_count": len(queries),
                    "cache_dir": str(prepared.cache_dir),
                    "rerank": rerank,
                    "modes": {},
                }
                for mode in modes:
                    mode_provider = None if mode == "bm25" else provider
                    mode_report = evaluate_mode(
                        queries,
                        store=store,
                        mode=mode,  # type: ignore[arg-type]
                        provider=mode_provider,
                        settings=settings,
                        collection="benchmark",
                        top_k=top_k,
                        rerank=rerank,
                    )
                    from custom_rag.eval.runner import _mode_payload

                    report["modes"][mode] = _mode_payload(mode_report)
                    m = report["modes"][mode]
                    print(
                        f"  {mode}: hit={m['hit_at_k']:.3f} ndcg={m['ndcg_at_k']:.3f}",
                        flush=True,
                    )
                report["comparison"] = build_comparison(report["modes"])
                write_report(report, out)
                print(f"wrote {out}", flush=True)
        finally:
            store.close()


if __name__ == "__main__":
    main()
