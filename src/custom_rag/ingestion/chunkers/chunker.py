"""Parent-child document chunking orchestrator."""

from __future__ import annotations

from custom_rag.core.types import (
    BlockType,
    Chunk,
    ChunkedDocument,
    ChunkRole,
    ContentBlock,
    ParsedDocument,
)
from custom_rag.ingestion.chunkers.assembler import assemble_parent_text
from custom_rag.ingestion.chunkers.config import ChunkConfig
from custom_rag.ingestion.chunkers.roles import (
    is_child_type,
    is_parent_type,
    nearest_parent_block,
    owning_parent_ids,
)
from custom_rag.ingestion.chunkers.splitter import split_text

_SYNTHETIC_PARENT_BLOCK_ID = "__synthetic_document__"


def chunk_document(
    document: ParsedDocument,
    *,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    active_config = config or ChunkConfig()
    blocks_by_id = {block.block_id: block for block in document.blocks}
    owning_ids = owning_parent_ids(document, blocks_by_id)

    chunks: list[Chunk] = []
    source_block_ids: list[str] = []
    parent_chunk_by_block_id: dict[str, str] = {}
    order = 0

    for block in document.blocks:
        if not is_parent_type(block.block_type) or block.block_id not in owning_ids:
            continue
        chunk_id = f"parent_{block.block_id}"
        parent_chunk_by_block_id[block.block_id] = chunk_id
        chunks.append(
            _build_parent_chunk(
                block=block,
                chunk_id=chunk_id,
                document=document,
                blocks_by_id=blocks_by_id,
                parent_max_chars=active_config.parent_max_chars,
                order=order,
            )
        )
        order += 1

    if not parent_chunk_by_block_id and any(
        is_child_type(block.block_type) and block.text for block in document.blocks
    ):
        synthetic = _synthetic_parent_chunk(document, active_config.parent_max_chars, order)
        chunks.append(synthetic)
        parent_chunk_by_block_id[_SYNTHETIC_PARENT_BLOCK_ID] = synthetic.chunk_id
        order += 1

    for block in document.blocks:
        if not is_child_type(block.block_type) or not block.text:
            continue
        source_block_ids.append(block.block_id)
        parts = split_text(
            block.text,
            max_chars=active_config.child_max_chars,
            overlap_chars=active_config.child_overlap_chars,
            min_chars=active_config.min_child_chars,
        )
        is_split = len(parts) > 1
        for index, part_text in enumerate(parts):
            split_index = 0 if not is_split else index + 1
            chunk_id = (
                f"child_{block.block_id}"
                if not is_split
                else f"child_{block.block_id}_part_{index + 1}"
            )
            parent_chunk_id = _resolve_parent_chunk_id(
                document=document,
                block=block,
                blocks_by_id=blocks_by_id,
                parent_chunk_by_block_id=parent_chunk_by_block_id,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    role=ChunkRole.CHILD,
                    text=part_text,
                    source_block_id=block.block_id,
                    parent_chunk_id=parent_chunk_id,
                    split_index=split_index,
                    block_type=block.block_type,
                    order=order,
                    hierarchy_path=list(block.hierarchy_path),
                    location=block.location,
                    metadata={
                        **block.metadata,
                        "char_count": len(part_text),
                        "is_split": is_split,
                    },
                )
            )
            order += 1

    return ChunkedDocument(
        metadata=document.metadata,
        chunks=chunks,
        source_block_ids=source_block_ids,
    )


def _build_parent_chunk(
    *,
    block: ContentBlock,
    chunk_id: str,
    document: ParsedDocument,
    blocks_by_id: dict[str, ContentBlock],
    parent_max_chars: int,
    order: int,
) -> Chunk:
    text = assemble_parent_text(
        document,
        block,
        blocks_by_id,
        parent_max_chars=parent_max_chars,
    )
    return Chunk(
        chunk_id=chunk_id,
        role=ChunkRole.PARENT,
        text=text,
        source_block_id=block.block_id,
        split_index=0,
        block_type=block.block_type,
        order=order,
        hierarchy_path=list(block.hierarchy_path),
        location=block.location,
        metadata={
            **block.metadata,
            "char_count": len(text),
            "is_split": False,
        },
    )


def _synthetic_parent_chunk(
    document: ParsedDocument,
    parent_max_chars: int,
    order: int,
) -> Chunk:
    if document.raw_text:
        text = document.raw_text
    else:
        leaf_texts = [
            block.text
            for block in document.blocks
            if is_child_type(block.block_type) and block.text
        ]
        text = "\n\n".join(leaf_texts)

    if len(text) > parent_max_chars:
        marker = "\n\n[...]"
        budget = max(parent_max_chars - len(marker), 0)
        text = text[:budget].rstrip() + marker

    parent_block_type = next(
        (item.block_type for item in document.blocks if is_parent_type(item.block_type)),
        document.blocks[0].block_type if document.blocks else BlockType.DOCUMENT,
    )

    return Chunk(
        chunk_id="parent_doc_synthetic",
        role=ChunkRole.PARENT,
        text=text,
        source_block_id=_SYNTHETIC_PARENT_BLOCK_ID,
        split_index=0,
        block_type=parent_block_type,
        order=order,
        hierarchy_path=["doc"],
        metadata={"char_count": len(text), "is_split": False, "synthetic": True},
    )


def _resolve_parent_chunk_id(
    *,
    document: ParsedDocument,
    block: ContentBlock,
    blocks_by_id: dict[str, ContentBlock],
    parent_chunk_by_block_id: dict[str, str],
) -> str | None:
    ancestor = nearest_parent_block(document, block, blocks_by_id)
    if ancestor is not None:
        mapped = parent_chunk_by_block_id.get(ancestor.block_id)
        if mapped is not None:
            return mapped
    return parent_chunk_by_block_id.get(_SYNTHETIC_PARENT_BLOCK_ID)
