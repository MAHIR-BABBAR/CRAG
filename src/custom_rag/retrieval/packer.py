"""Char-budget context packing with citation markers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from custom_rag.storage.base import VectorHit

_TRUNCATE_MARK = "…"


@dataclass(frozen=True)
class Citation:
    index: int
    doc_id: str
    chunk_id: str
    parent_chunk_id: str | None
    source_path: str
    score: float
    page: int | None = None
    slide: int | None = None


@dataclass(frozen=True)
class PackedContext:
    text: str
    citations: list[Citation]
    truncated: bool
    chars_used: int


def assemble_context(
    hits: list[VectorHit],
    *,
    max_chars: int,
    include_parents: bool = True,
    source_path_for: Callable[[str], str | None] | None = None,
) -> PackedContext:
    """Pack ranked hits into a char-budgeted prompt block with ``[n]`` citations.

    Prefers parent text when available and ``include_parents`` is true.
    Dedupes by ``(doc_id, parent_chunk_id or chunk_id)``.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    blocks: list[str] = []
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    truncated = False
    used = 0
    cite_index = 0

    for hit in hits:
        parent_id = hit.chunk.parent_chunk_id
        dedupe_key = (hit.doc_id, parent_id or hit.chunk_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if include_parents and hit.parent is not None and hit.parent.text.strip():
            body = hit.parent.text.strip()
        else:
            body = hit.chunk.text.strip()
        if not body:
            continue

        source_path = ""
        if source_path_for is not None:
            source_path = source_path_for(hit.doc_id) or ""
        if not source_path:
            source_path = hit.doc_id

        cite_index += 1
        header = f"[{cite_index}] ({source_path})\n"
        trailer = "\n\n"
        overhead = len(header) + len(trailer)
        remaining = max_chars - used
        if remaining <= overhead:
            truncated = True
            break

        available_for_body = remaining - overhead
        block_truncated = False
        if len(body) > available_for_body:
            keep = max(0, available_for_body - len(_TRUNCATE_MARK))
            body = body[:keep] + _TRUNCATE_MARK if keep > 0 else _TRUNCATE_MARK
            block_truncated = True
            truncated = True

        block = f"{header}{body}{trailer}"
        blocks.append(block)
        used += len(block)

        location = hit.chunk.location
        page = location.page_number if location is not None else None
        slide = location.slide_number if location is not None else None
        citations.append(
            Citation(
                index=cite_index,
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                parent_chunk_id=parent_id,
                source_path=source_path,
                score=hit.score,
                page=page,
                slide=slide,
            )
        )
        if block_truncated:
            break

    text = "".join(blocks).rstrip()
    return PackedContext(
        text=text,
        citations=citations,
        truncated=truncated,
        chars_used=len(text),
    )
