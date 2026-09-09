"""Download and cache public IR benchmarks for offline eval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCIFACT_DATASET_ID = "beir/scifact/test"


@dataclass(frozen=True)
class PreparedCorpus:
    name: str
    cache_dir: Path
    docs_dir: Path
    qrels_path: Path
    manifest_path: Path
    doc_count: int
    query_count: int
    dataset_id: str = SCIFACT_DATASET_ID
    fingerprint: str = ""
    subset: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_id": self.dataset_id,
            "doc_count": self.doc_count,
            "query_count": self.query_count,
            "fingerprint": self.fingerprint,
            "subset": self.subset,
        }


def default_eval_cache_dir() -> Path:
    return Path("workspace_index") / "eval_cache"


def corpus_fingerprint(dataset_id: str, max_docs: int | None, max_queries: int | None) -> str:
    """Identify what a cache directory actually holds.

    A subset built for a smoke run and the full corpus land in the same cache
    directory. Without this, a 100-document smoke run silently becomes the
    corpus that every later full run reports numbers against.
    """
    payload = json.dumps(
        {"dataset_id": dataset_id, "max_docs": max_docs, "max_queries": max_queries},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_doc_filename(doc_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in doc_id)


def _write_doc(docs_dir: Path, doc: Any) -> str | None:
    doc_id = str(doc.doc_id)
    title = getattr(doc, "title", "") or ""
    text = getattr(doc, "text", "") or ""
    body = f"{title}\n\n{text}".strip() if title else text.strip()
    if not body:
        return None
    safe_id = _safe_doc_filename(doc_id)
    (docs_dir / f"{safe_id}.txt").write_text(body, encoding="utf-8")
    (docs_dir / f"{safe_id}.meta.json").write_text(
        json.dumps({"external_id": doc_id}),
        encoding="utf-8",
    )
    return doc_id


def prepare_scifact(
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    max_docs: int | None = None,
    max_queries: int | None = None,
) -> PreparedCorpus:
    """Download SciFact test (BEIR) via ir_datasets and write CRAG-ready cache files."""
    fingerprint = corpus_fingerprint(SCIFACT_DATASET_ID, max_docs, max_queries)
    subset = max_docs is not None or max_queries is not None
    # Subsets live beside the full corpus rather than on top of it, so a smoke
    # run cannot leave a truncated corpus behind for the next full run to use.
    default_root = default_eval_cache_dir() / (f"scifact-{fingerprint}" if subset else "scifact")
    root = cache_dir or default_root
    docs_dir = root / "docs"
    qrels_path = root / "qrels.json"
    manifest_path = root / "manifest.json"

    if (
        not force
        and manifest_path.is_file()
        and qrels_path.is_file()
        and docs_dir.is_dir()
        and any(docs_dir.glob("*.txt"))
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            return PreparedCorpus(
                name="scifact",
                cache_dir=root,
                docs_dir=docs_dir,
                qrels_path=qrels_path,
                manifest_path=manifest_path,
                doc_count=int(manifest.get("doc_count", 0)),
                query_count=int(manifest.get("query_count", 0)),
                fingerprint=fingerprint,
                subset=subset,
            )

    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            'ir-datasets is required for SciFact eval; install with: pip install "custom-rag[eval]"'
        ) from exc

    # Parent ``beir/scifact`` has docs/queries only; test split includes qrels.
    dataset = ir_datasets.load(SCIFACT_DATASET_ID)
    if docs_dir.exists():
        for stale in docs_dir.glob("*"):
            stale.unlink()
    docs_dir.mkdir(parents=True, exist_ok=True)

    qrel_map: dict[str, set[str]] = {}
    for qrel in dataset.qrels_iter():
        if int(getattr(qrel, "relevance", 0) or 0) <= 0:
            continue
        qrel_map.setdefault(str(qrel.query_id), set()).add(str(qrel.doc_id))

    queries_out: list[dict[str, Any]] = []
    needed_doc_ids: set[str] = set()
    for query in dataset.queries_iter():
        qid = str(query.query_id)
        relevant_ids = sorted(qrel_map.get(qid, set()))
        if not relevant_ids:
            continue
        queries_out.append(
            {
                "id": qid,
                "query": str(query.text),
                "focus": "benchmark",
                "relevant": [{"external_id": rid} for rid in relevant_ids],
            }
        )
        needed_doc_ids.update(relevant_ids)
        if max_queries is not None and len(queries_out) >= max_queries:
            break

    # Prefer qrel-covered docs so subset smokes remain labeled.
    docs_by_id: dict[str, Any] = {}
    filler: list[Any] = []
    filler_budget = None if max_docs is None else max(0, max_docs - len(needed_doc_ids))
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        if doc_id in needed_doc_ids:
            docs_by_id[doc_id] = doc
        elif filler_budget is None or len(filler) < filler_budget:
            filler.append(doc)
        if len(docs_by_id) >= len(needed_doc_ids) and (
            filler_budget is not None and len(filler) >= filler_budget
        ):
            break

    selected: list[Any] = [docs_by_id[did] for did in sorted(docs_by_id)]
    selected.extend(filler)

    doc_count = 0
    written_ids: set[str] = set()
    for doc in selected:
        written = _write_doc(docs_dir, doc)
        if written is None:
            continue
        written_ids.add(written)
        doc_count += 1

    # Drop queries whose relevant docs were not written (subset edge case).
    filtered_queries: list[dict[str, Any]] = []
    for entry in queries_out:
        relevant = entry["relevant"]
        if any(rule["external_id"] in written_ids for rule in relevant):
            filtered_queries.append(entry)

    query_count = len(filtered_queries)
    qrels_path.write_text(
        json.dumps(
            {
                "description": "SciFact test (BEIR) via ir_datasets — external_id qrels",
                "queries": filtered_queries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "name": "scifact",
        "ir_datasets_id": SCIFACT_DATASET_ID,
        "doc_count": doc_count,
        "query_count": query_count,
        "max_docs": max_docs,
        "max_queries": max_queries,
        "subset": subset,
        "fingerprint": fingerprint,
        "prepared_at": datetime.now(tz=UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PreparedCorpus(
        name="scifact",
        cache_dir=root,
        docs_dir=docs_dir,
        qrels_path=qrels_path,
        manifest_path=manifest_path,
        doc_count=doc_count,
        query_count=query_count,
        fingerprint=fingerprint,
        subset=subset,
    )
