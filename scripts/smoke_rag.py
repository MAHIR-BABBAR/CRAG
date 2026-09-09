"""End-to-end smoke run over the bundled fixtures, for manual sanity checks."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from custom_rag.core.config import CRAGSettings, EmbeddingSettings, StorageSettings
from custom_rag.core.providers.mock import MockEmbeddingProvider
from custom_rag.eval.runner import default_fixture_paths, run_eval_suite
from custom_rag.ingestion import index_file
from custom_rag.retrieval import assemble_context, retrieve_with_parents, score_retrieval
from custom_rag.storage.sqlite_store import SQLiteIndexStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
QRELS = ROOT / "tests" / "eval" / "qrels.json"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="crag-smoke-") as tmp:
        db_path = Path(tmp) / "smoke.sqlite3"
        settings = CRAGSettings(
            embedding=EmbeddingSettings(provider="mock", model="mock-embed-v1"),
            storage=StorageSettings(path=str(db_path), collection="smoke"),
        )
        provider = MockEmbeddingProvider()
        with SQLiteIndexStore(db_path, default_collection="smoke") as store:
            for path in default_fixture_paths(FIXTURES):
                index_file(path, provider=provider, store=store, collection="smoke")
            result = retrieve_with_parents(
                "Install the package with pip",
                store=store,
                provider=provider,
                mode="hybrid",
                top_k=5,
                collection="smoke",
                settings=settings,
            )
            packed = assemble_context(
                result.hits,
                max_chars=2000,
                source_path_for=store.get_source_path,
            )
            quality = score_retrieval(result.hits, packed, mode=result.mode, min_score_high=0.0)
            print(f"hits={len(result.hits)} quality={quality.label}")
            print(packed.text[:400] or "(empty)")

        report = run_eval_suite(
            qrels_path=QRELS,
            fixtures_dir=FIXTURES,
            db_path=Path(tmp) / "eval.sqlite3",
            provider_name="mock",
            top_k=5,
        )
        print("--- eval ---")
        for mode, payload in report["modes"].items():
            print(
                f"{mode}: hit@k={payload['hit_at_k']:.3f} "
                f"recall@k={payload['recall_at_k']:.3f} mrr={payload['mrr']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
