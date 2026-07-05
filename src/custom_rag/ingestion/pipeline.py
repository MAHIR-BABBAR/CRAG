"""Stage 1 ingestion orchestrator."""

from __future__ import annotations

from pathlib import Path

from custom_rag.core.types import DocumentMetadata, ParsedDocument
from custom_rag.ingestion.metadata.file_metadata import extract_file_metadata
from custom_rag.ingestion.parsers.composite import build_default_registry
from custom_rag.ingestion.parsers.registry import ParserRegistry

_DEFAULT_REGISTRY = build_default_registry()


def parse_file(path: Path | str, *, registry: ParserRegistry | None = None) -> ParsedDocument:
    resolved = Path(path).expanduser().resolve()
    active_registry = registry or _DEFAULT_REGISTRY

    file_metadata = extract_file_metadata(resolved)
    parser = active_registry.get_parser(resolved, file_metadata.mime_type)
    document = parser.parse(resolved, file_metadata)
    return _merge_file_metadata(document, file_metadata)


def _merge_file_metadata(
    document: ParsedDocument, file_metadata: DocumentMetadata
) -> ParsedDocument:
    merged_extra = {**file_metadata.extra, **document.metadata.extra}
    if document.metadata.encoding:
        encoding = document.metadata.encoding
    else:
        encoding = file_metadata.encoding

    merged = file_metadata.model_copy(
        update={
            "doc_type": document.metadata.doc_type or file_metadata.doc_type,
            "language": document.metadata.language or file_metadata.language,
            "encoding": encoding,
            "extra": merged_extra,
        }
    )
    return document.model_copy(update={"metadata": merged})
