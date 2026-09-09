"""Block-type to chunk-role mapping helpers."""

from __future__ import annotations

from custom_rag.core.types import BlockType, ContentBlock, ParsedDocument

PARENT_TYPES: frozenset[BlockType] = frozenset(
    {
        BlockType.DOCUMENT,
        BlockType.PAGE,
        BlockType.SLIDE,
        BlockType.SECTION,
        BlockType.MODULE,
        BlockType.CODE_CONTAINER,
        BlockType.TABLE,
    }
)

CHILD_TYPES: frozenset[BlockType] = frozenset(
    {
        BlockType.PARAGRAPH,
        BlockType.CODE_BLOCK,
        BlockType.CODE_SYMBOL,
        BlockType.TABLE_ROW,
        BlockType.LIST_ITEM,
        BlockType.SHAPE,
        BlockType.NOTE,
        BlockType.FIGURE,
        BlockType.RECORD,
    }
)


def is_parent_type(block_type: BlockType) -> bool:
    return block_type in PARENT_TYPES


def is_child_type(block_type: BlockType) -> bool:
    return block_type in CHILD_TYPES


def nearest_parent_block(
    document: ParsedDocument,
    block: ContentBlock,
    blocks_by_id: dict[str, ContentBlock],
) -> ContentBlock | None:
    current_id = block.parent_block_id
    while current_id is not None:
        ancestor = blocks_by_id.get(current_id)
        if ancestor is None:
            return None
        if is_parent_type(ancestor.block_type):
            return ancestor
        current_id = ancestor.parent_block_id
    return None


def owned_leaf_blocks(
    document: ParsedDocument,
    parent_block_id: str,
    blocks_by_id: dict[str, ContentBlock],
) -> list[ContentBlock]:
    """Return leaf blocks owned by this parent (stop at nested parent-type nodes)."""
    leaves: list[ContentBlock] = []

    def walk(block_id: str) -> None:
        for child in document.children_of(block_id):
            if is_parent_type(child.block_type):
                continue
            if is_child_type(child.block_type) and child.text:
                leaves.append(child)
            elif child.block_id in blocks_by_id:
                walk(child.block_id)

    walk(parent_block_id)
    return sorted(leaves, key=lambda block: block.order)


def descendant_leaf_blocks(
    document: ParsedDocument,
    parent_block_id: str,
    blocks_by_id: dict[str, ContentBlock],
) -> list[ContentBlock]:
    """All leaf descendants, including those under nested parents (legacy helper)."""
    leaves: list[ContentBlock] = []

    def walk(block_id: str) -> None:
        for child in document.children_of(block_id):
            if is_child_type(child.block_type) and child.text:
                leaves.append(child)
            if child.block_id in blocks_by_id:
                walk(child.block_id)

    walk(parent_block_id)
    return sorted(leaves, key=lambda block: block.order)


def owning_parent_ids(
    document: ParsedDocument,
    blocks_by_id: dict[str, ContentBlock],
) -> set[str]:
    """Parent block IDs that are the nearest parent of at least one leaf."""
    owners: set[str] = set()
    for block in document.blocks:
        if not is_child_type(block.block_type) or not block.text:
            continue
        owner = nearest_parent_block(document, block, blocks_by_id)
        if owner is not None:
            owners.add(owner.block_id)
    return owners


def has_descendant_leaf(
    document: ParsedDocument,
    parent_block_id: str,
    blocks_by_id: dict[str, ContentBlock],
) -> bool:
    return bool(descendant_leaf_blocks(document, parent_block_id, blocks_by_id))
