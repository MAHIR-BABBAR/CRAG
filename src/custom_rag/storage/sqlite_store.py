"""SQLite-backed document and vector index."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from custom_rag.core.exceptions import StorageError
from custom_rag.core.types import (
    BlockLocation,
    BlockType,
    Chunk,
    ChunkRole,
    EmbeddedDocument,
)
from custom_rag.storage.base import VectorHit
from custom_rag.storage.schema import SCHEMA_SQL, storage_id
from custom_rag.storage.vectors import cosine_similarity, pack_vector, unpack_vector

_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_BUSY_TIMEOUT_MS = 5000


def sanitize_fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free text.

    Tokens are OR-combined so long claims (e.g. SciFact) still recall partial
    lexical overlaps. Quoted tokens prevent operator injection. Returns an
    empty string when no usable tokens remain.
    """
    # Drop 1-char noise; cap length to keep MATCH plans bounded.
    tokens = [t for t in _FTS_TOKEN_RE.findall(query) if len(t) >= 2][:64]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


class SQLiteIndexStore:
    """Local index store with content-hash dedup support."""

    def __init__(
        self,
        path: Path | str,
        *,
        default_collection: str = "default",
        dense_search: str = "python",
    ) -> None:
        self.path = Path(path)
        self.default_collection = default_collection
        self.dense_search = dense_search if dense_search in ("python", "numpy") else "python"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._signature_cache: dict[str, list[tuple[str, str, int]]] = {}
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            # Some filesystems and hosts reject WAL; fall back to the default journal.
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"failed to open index at {self.path}: {exc}", cause=exc) from exc

    def __enter__(self) -> SQLiteIndexStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ping(self) -> None:
        """Cheap reachability check for health endpoints."""
        with self._lock:
            try:
                self._conn.execute("SELECT 1").fetchone()
            except sqlite3.Error as exc:
                raise StorageError(f"index unreachable at {self.path}: {exc}", cause=exc) from exc

    def get_indexed_counts(self, source_path: str) -> tuple[str, int, int] | None:
        """Return ``(doc_id, child_count, parent_count)`` for a source path."""
        with self._lock:
            row = self._conn.execute(
                "SELECT doc_id FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if row is None:
                return None
            doc_id = str(row["doc_id"])
            children = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND role = 'child'",
                (doc_id,),
            ).fetchone()["n"]
            parents = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND role = 'parent'",
                (doc_id,),
            ).fetchone()["n"]
            return doc_id, int(children), int(parents)

    def embedding_signatures(self, collection: str | None = None) -> list[tuple[str, str, int]]:
        """Distinct ``(provider, model, dimensions)`` triples in a collection.

        Vectors from different models are not comparable, so callers use this to
        refuse a query embedded with something other than what was indexed.
        """
        active_collection = collection or self.default_collection
        cached = self._signature_cache.get(active_collection)
        if cached is not None:
            return cached
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT e.provider, e.model, e.dimensions
                FROM embeddings e
                JOIN documents d ON d.doc_id = e.doc_id
                WHERE d.collection = ?
                """,
                (active_collection,),
            ).fetchall()
        signatures = [
            (str(row["provider"]), str(row["model"]), int(row["dimensions"])) for row in rows
        ]
        self._signature_cache[active_collection] = signatures
        return signatures

    def get_source_path(self, doc_id: str) -> str | None:
        """Return the source_path for a document id, if present."""
        with self._lock:
            row = self._conn.execute(
                "SELECT source_path FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return None if row is None else str(row["source_path"])

    def get_external_id(self, doc_id: str) -> str | None:
        """Return ``extra.external_id`` for a document, if present."""
        with self._lock:
            row = self._conn.execute(
                "SELECT extra_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                extra = json.loads(row["extra_json"] or "{}")
            except json.JSONDecodeError:
                return None
            value = extra.get("external_id")
            return None if value is None else str(value)

    def get_document_text(self, doc_id: str) -> str:
        """Concatenated child text for a document, in chunk order.

        Used by evaluation to judge relevance against the whole document rather
        than against whichever chunk the ranker happened to return.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT text FROM chunks
                WHERE doc_id = ? AND role = 'child'
                ORDER BY "order"
                """,
                (doc_id,),
            ).fetchall()
        return "\n".join(str(row["text"]) for row in rows)

    def get_content_hash(self, source_path: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            return None if row is None else str(row["content_hash"])

    def set_content_hash(self, source_path: str, content_hash: str) -> None:
        # Hash is written atomically during upsert; keep protocol compatibility.
        _ = (source_path, content_hash)

    def upsert(self, document: EmbeddedDocument, *, collection: str | None = None) -> None:
        if document.skipped:
            return

        active_collection = collection or self.default_collection
        self._signature_cache.pop(active_collection, None)
        metadata = document.metadata
        source_path = metadata.source_path
        indexed_at = datetime.now(tz=UTC)
        doc_id = metadata.doc_id

        with self._lock:
            try:
                with self._conn:
                    existing = self._conn.execute(
                        "SELECT doc_id FROM documents WHERE source_path = ?",
                        (source_path,),
                    ).fetchone()
                    if existing is not None:
                        old_doc_id = str(existing["doc_id"])
                        self._delete_document(old_doc_id)

                    self._conn.execute(
                        """
                        INSERT INTO documents (
                            doc_id, source_path, source_uri, doc_type, mime_type,
                            content_hash, file_size, modified_at, indexed_at,
                            language, encoding, collection, extra_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            source_path,
                            metadata.source_uri,
                            metadata.doc_type,
                            metadata.mime_type,
                            metadata.content_hash,
                            metadata.file_size,
                            metadata.modified_at.isoformat(),
                            indexed_at.isoformat(),
                            metadata.language,
                            metadata.encoding,
                            active_collection,
                            json.dumps(metadata.extra),
                        ),
                    )

                    for parent in document.parents:
                        self._insert_chunk(doc_id, parent)

                    for embedded in document.children:
                        self._insert_chunk(doc_id, embedded.chunk)
                        sid = storage_id(doc_id, embedded.chunk.chunk_id)
                        self._conn.execute(
                            """
                            INSERT INTO embeddings (
                                storage_id, doc_id, chunk_id, vector, dimensions,
                                model, provider, text_hash, embedded_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                sid,
                                doc_id,
                                embedded.chunk.chunk_id,
                                pack_vector(embedded.embedding),
                                embedded.dimensions,
                                embedded.model,
                                embedded.provider,
                                embedded.text_hash,
                                embedded.embedded_at.isoformat(),
                            ),
                        )
                    self._sync_fts_for_doc(doc_id)
            except sqlite3.Error as exc:
                raise StorageError(f"failed to upsert document {doc_id}: {exc}", cause=exc) from exc

    def search_vector(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[VectorHit]:
        active_collection = collection or self.default_collection
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.doc_id, e.chunk_id, e.vector, c.role, c.text, c.source_block_id,
                       c.parent_chunk_id, c.split_index, c.block_type, c."order",
                       c.hierarchy_path_json, c.location_json, c.metadata_json
                FROM embeddings e
                JOIN chunks c ON c.storage_id = e.storage_id
                JOIN documents d ON d.doc_id = e.doc_id
                WHERE d.collection = ?
                """,
                (active_collection,),
            ).fetchall()

        if not rows:
            return []

        if self.dense_search == "numpy":
            return self._search_vector_numpy(query_embedding, rows, top_k=top_k)

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            vector = unpack_vector(bytes(row["vector"]))
            try:
                score = cosine_similarity(query_embedding, vector)
            except ValueError:
                continue
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[VectorHit] = []
        for score, row in scored[:top_k]:
            chunk = _row_to_chunk(row)
            hits.append(
                VectorHit(
                    doc_id=str(row["doc_id"]),
                    chunk_id=chunk.chunk_id,
                    score=score,
                    chunk=chunk,
                    parent=None,
                )
            )
        return hits

    def _search_vector_numpy(
        self,
        query_embedding: list[float],
        rows: list[sqlite3.Row],
        *,
        top_k: int,
    ) -> list[VectorHit]:
        from custom_rag.storage.vectors import batch_cosine_topk

        try:
            import numpy as np
        except ImportError as exc:
            raise StorageError(
                'dense_search="numpy" requires numpy; install with: pip install "custom-rag[numpy]"'
            ) from exc

        vectors: list[list[float]] = []
        valid_rows: list[sqlite3.Row] = []
        dim = len(query_embedding)
        for row in rows:
            try:
                vector = unpack_vector(bytes(row["vector"]))
            except ValueError:
                continue
            if len(vector) != dim:
                continue
            vectors.append(vector)
            valid_rows.append(row)
        if not vectors:
            return []
        matrix = np.asarray(vectors, dtype=np.float32)
        ranked = batch_cosine_topk(query_embedding, matrix, top_k=top_k)
        hits: list[VectorHit] = []
        for index, score in ranked:
            row = valid_rows[index]
            chunk = _row_to_chunk(row)
            hits.append(
                VectorHit(
                    doc_id=str(row["doc_id"]),
                    chunk_id=chunk.chunk_id,
                    score=score,
                    chunk=chunk,
                    parent=None,
                )
            )
        return hits

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[VectorHit]:
        active_collection = collection or self.default_collection
        match_query = sanitize_fts_query(query)
        if not match_query:
            return []

        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT c.doc_id, c.chunk_id, c.role, c.text, c.source_block_id,
                           c.parent_chunk_id, c.split_index, c.block_type, c."order",
                           c.hierarchy_path_json, c.location_json, c.metadata_json,
                           bm25(chunks_fts) AS bm25_score
                    FROM chunks_fts
                    JOIN chunks c ON c.storage_id = chunks_fts.storage_id
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE chunks_fts MATCH ?
                      AND d.collection = ?
                    ORDER BY bm25_score ASC
                    LIMIT ?
                    """,
                    (match_query, active_collection, top_k),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                # A malformed MATCH expression is a bug in sanitize_fts_query, and
                # a missing/corrupt FTS table is an index problem. Neither is an
                # empty result set, so neither should look like one.
                raise StorageError(
                    f"BM25 search failed for collection {active_collection!r}: {exc}",
                    cause=exc,
                ) from exc

        hits: list[VectorHit] = []
        for row in rows:
            chunk = _row_to_chunk(row)
            hits.append(
                VectorHit(
                    doc_id=str(row["doc_id"]),
                    chunk_id=chunk.chunk_id,
                    score=float(row["bm25_score"]),
                    chunk=chunk,
                    parent=None,
                )
            )
        return hits

    def get_chunk(self, doc_id: str, chunk_id: str) -> Chunk | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT chunk_id, role, text, source_block_id, parent_chunk_id, split_index,
                       block_type, "order", hierarchy_path_json, location_json, metadata_json
                FROM chunks
                WHERE doc_id = ? AND chunk_id = ?
                """,
                (doc_id, chunk_id),
            ).fetchone()
            return None if row is None else _row_to_chunk(row)

    def get_parent(self, doc_id: str, parent_chunk_id: str) -> Chunk | None:
        return self.get_chunk(doc_id, parent_chunk_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _delete_document(self, doc_id: str) -> None:
        self._delete_fts_for_doc(doc_id)
        self._conn.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def _delete_fts_for_doc(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))

    def _sync_fts_for_doc(self, doc_id: str) -> None:
        self._delete_fts_for_doc(doc_id)
        rows = self._conn.execute(
            """
            SELECT storage_id, doc_id, chunk_id, text
            FROM chunks
            WHERE doc_id = ? AND role = 'child'
            """,
            (doc_id,),
        ).fetchall()
        for row in rows:
            self._conn.execute(
                """
                INSERT INTO chunks_fts (text, storage_id, doc_id, chunk_id)
                VALUES (?, ?, ?, ?)
                """,
                (row["text"], row["storage_id"], row["doc_id"], row["chunk_id"]),
            )

    def _rebuild_fts(self) -> None:
        """Rebuild the entire FTS index (test/maintenance helper)."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks_fts")
            rows = self._conn.execute(
                """
                SELECT storage_id, doc_id, chunk_id, text
                FROM chunks
                WHERE role = 'child'
                """
            ).fetchall()
            for row in rows:
                self._conn.execute(
                    """
                    INSERT INTO chunks_fts (text, storage_id, doc_id, chunk_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (row["text"], row["storage_id"], row["doc_id"], row["chunk_id"]),
                )

    def _insert_chunk(self, doc_id: str, chunk: Chunk) -> None:
        sid = storage_id(doc_id, chunk.chunk_id)
        location_json = None if chunk.location is None else chunk.location.model_dump_json()
        self._conn.execute(
            """
            INSERT INTO chunks (
                storage_id, doc_id, chunk_id, role, text, source_block_id,
                parent_chunk_id, split_index, block_type, "order",
                hierarchy_path_json, location_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                doc_id,
                chunk.chunk_id,
                chunk.role.value,
                chunk.text,
                chunk.source_block_id,
                chunk.parent_chunk_id,
                chunk.split_index,
                chunk.block_type.value,
                chunk.order,
                json.dumps(chunk.hierarchy_path),
                location_json,
                json.dumps(chunk.metadata),
            ),
        )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    location = None
    if row["location_json"]:
        location = BlockLocation.model_validate_json(row["location_json"])
    metadata: dict[str, Any] = json.loads(row["metadata_json"] or "{}")
    hierarchy_path = json.loads(row["hierarchy_path_json"] or "[]")
    return Chunk(
        chunk_id=str(row["chunk_id"]),
        role=ChunkRole(str(row["role"])),
        text=str(row["text"]),
        source_block_id=str(row["source_block_id"]),
        parent_chunk_id=row["parent_chunk_id"],
        split_index=int(row["split_index"]),
        block_type=BlockType(str(row["block_type"])),
        order=int(row["order"]),
        hierarchy_path=list(hierarchy_path),
        location=location,
        metadata=metadata,
    )
