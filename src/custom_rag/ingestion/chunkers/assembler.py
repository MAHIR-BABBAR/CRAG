"""Assemble parent chunk text from owned leaf blocks only."""

from __future__ import annotations

from custom_rag.core.types import ContentBlock, ParsedDocument
from custom_rag.ingestion.chunkers.roles import owned_leaf_blocks

_TRUNCATION_MARKER = "\n\n[...]"


def assemble_parent_text(
    document: ParsedDocument,
    parent_block: ContentBlock,
    blocks_by_id: dict[str, ContentBlock],
    *,
    parent_max_chars: int,
) -> str:
    """Build parent context from local parent text + owned leaves only.

    Nested parent-type subtrees are excluded so each leaf appears in at most
    one parent assembly (nearest-parent ownership).
    """
    sections: list[str] = []
    if parent_block.text:
        sections.append(parent_block.text)

    for leaf in owned_leaf_blocks(document, parent_block.block_id, blocks_by_id):
        sections.append(leaf.text)

    assembled = "\n\n".join(sections)
    if len(assembled) <= parent_max_chars:
        return assembled

    if parent_max_chars <= len(_TRUNCATION_MARKER):
        return assembled[:parent_max_chars]

    budget = parent_max_chars - len(_TRUNCATION_MARKER)
    return assembled[:budget].rstrip() + _TRUNCATION_MARKER
