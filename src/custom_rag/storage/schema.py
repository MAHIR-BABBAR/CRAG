"""SQLite schema DDL for the local CRAG index."""

from __future__ import annotations

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    source_uri TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    mime_type TEXT,
    content_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    language TEXT,
    encoding TEXT,
    collection TEXT NOT NULL DEFAULT 'default',
    extra_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    storage_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    source_block_id TEXT NOT NULL,
    parent_chunk_id TEXT,
    split_index INTEGER NOT NULL DEFAULT 0,
    block_type TEXT NOT NULL,
    "order" INTEGER NOT NULL DEFAULT 0,
    hierarchy_path_json TEXT NOT NULL DEFAULT '[]',
    location_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (doc_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    storage_id TEXT PRIMARY KEY REFERENCES chunks(storage_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    vector BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedded_at TEXT NOT NULL,
    UNIQUE (doc_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_role ON chunks(role);
CREATE INDEX IF NOT EXISTS idx_embeddings_doc_id ON embeddings(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    storage_id UNINDEXED,
    doc_id UNINDEXED,
    chunk_id UNINDEXED
);
"""


def storage_id(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}::{chunk_id}"
