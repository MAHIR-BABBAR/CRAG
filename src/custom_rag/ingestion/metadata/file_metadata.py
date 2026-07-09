"""File-level metadata extraction."""

from __future__ import annotations

import mimetypes
import uuid
from datetime import UTC, datetime
from pathlib import Path

from custom_rag.core.exceptions import ParseFailedError
from custom_rag.core.types import DocumentMetadata
from custom_rag.ingestion.parsers.base import content_hash


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def detect_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type


def extract_file_metadata(path: Path, *, doc_type: str | None = None) -> DocumentMetadata:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ParseFailedError(str(resolved), "path is not a file")

    try:
        raw = resolved.read_bytes()
        stat = resolved.stat()
    except OSError as exc:
        raise ParseFailedError(str(resolved), "unable to read file metadata", cause=exc) from exc

    resolved_doc_type = doc_type or resolved.suffix.lstrip(".").lower() or "unknown"
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

    return DocumentMetadata(
        doc_id=str(uuid.uuid4()),
        source_path=str(resolved),
        source_uri=file_uri(resolved),
        doc_type=resolved_doc_type,
        mime_type=detect_mime_type(resolved),
        content_hash=content_hash(raw),
        file_size=stat.st_size,
        modified_at=modified_at,
    )
