"""Map Unstructured API elements to CRAG ContentBlock hierarchy."""

from __future__ import annotations

from typing import Any

from custom_rag.core.types import BlockLocation, BlockType, ContentBlock

_SKIP_TYPES = frozenset({"Footer", "Header"})
_TITLE_TYPES = frozenset({"Title"})
_HEADING_TYPES = frozenset({"Header"})
_LIST_TYPES = frozenset({"ListItem"})
_TABLE_TYPES = frozenset({"Table"})
_FIGURE_TYPES = frozenset({"Image", "FigureCaption"})


def map_elements_to_blocks(elements: list[dict[str, Any]]) -> list[ContentBlock]:
    """Convert Unstructured partition elements into page-parented ContentBlocks."""
    blocks: list[ContentBlock] = []
    page_blocks: dict[int, str] = {}
    section_stack: dict[int, list[tuple[str, int]]] = {}
    counters: dict[str, int] = {}

    for element in elements:
        element_type = str(element.get("type", ""))
        text = str(element.get("text", "")).strip()
        if not text or element_type in _SKIP_TYPES:
            continue

        meta = element.get("metadata") or {}
        page_number = _page_number(meta)
        location = _block_location(meta)
        page_id = _ensure_page_block(blocks, page_blocks, page_number)

        if element_type in _TITLE_TYPES or element_type in _HEADING_TYPES:
            _add_section_block(
                blocks=blocks,
                counters=counters,
                section_stack=section_stack,
                page_number=page_number,
                page_id=page_id,
                text=text,
                element_type=element_type,
                location=location,
                meta=meta,
            )
            continue

        parent_id, hierarchy = _current_parent(page_id, page_number, section_stack)

        if element_type in _TABLE_TYPES:
            _add_table_blocks(
                blocks=blocks,
                counters=counters,
                parent_id=parent_id,
                hierarchy=hierarchy,
                text=text,
                location=location,
                meta=meta,
                element_type=element_type,
            )
            continue

        block_type, block_id, order_key = _child_block_spec(element_type, counters)
        blocks.append(
            ContentBlock(
                block_id=block_id,
                block_type=block_type,
                text=text,
                parent_block_id=parent_id,
                order=counters[order_key],
                hierarchy_path=[*hierarchy, block_id],
                location=location,
                metadata=_element_metadata(element_type, meta),
            )
        )

    return blocks


def accepted_element_texts(elements: list[dict[str, Any]]) -> list[str]:
    """Return accepted element texts in source order for ``raw_text``."""
    texts: list[str] = []
    for element in elements:
        element_type = str(element.get("type", ""))
        text = str(element.get("text", "")).strip()
        if not text or element_type in _SKIP_TYPES:
            continue
        texts.append(text)
    return texts


def _child_block_spec(element_type: str, counters: dict[str, int]) -> tuple[BlockType, str, str]:
    if element_type in _FIGURE_TYPES:
        counters["figure"] = counters.get("figure", 0) + 1
        return BlockType.FIGURE, f"fig_{counters['figure']}", "figure"
    if element_type in _LIST_TYPES:
        counters["list"] = counters.get("list", 0) + 1
        return BlockType.LIST_ITEM, f"li_{counters['list']}", "list"
    counters["paragraph"] = counters.get("paragraph", 0) + 1
    return BlockType.PARAGRAPH, f"p_{counters['paragraph']}", "paragraph"


def _add_table_blocks(
    *,
    blocks: list[ContentBlock],
    counters: dict[str, int],
    parent_id: str,
    hierarchy: list[str],
    text: str,
    location: BlockLocation | None,
    meta: dict[str, Any],
    element_type: str,
) -> None:
    counters["table"] = counters.get("table", 0) + 1
    table_id = f"tbl_{counters['table']}"
    row_id = f"{table_id}_row_0"
    blocks.append(
        ContentBlock(
            block_id=table_id,
            block_type=BlockType.TABLE,
            text="",
            parent_block_id=parent_id,
            order=counters["table"],
            hierarchy_path=[*hierarchy, table_id],
            location=location,
            metadata=_element_metadata(element_type, meta),
        )
    )
    blocks.append(
        ContentBlock(
            block_id=row_id,
            block_type=BlockType.TABLE_ROW,
            text=text,
            parent_block_id=table_id,
            order=0,
            hierarchy_path=[*hierarchy, table_id, row_id],
            location=location,
            metadata={"row_index": 0, **_element_metadata(element_type, meta)},
        )
    )


def _add_section_block(
    *,
    blocks: list[ContentBlock],
    counters: dict[str, int],
    section_stack: dict[int, list[tuple[str, int]]],
    page_number: int,
    page_id: str,
    text: str,
    element_type: str,
    location: BlockLocation | None,
    meta: dict[str, Any],
) -> None:
    counters["section"] = counters.get("section", 0) + 1
    block_id = f"sec_{counters['section']}"
    level = 1 if element_type in _TITLE_TYPES else 2

    stack = section_stack.setdefault(page_number, [])
    while stack and stack[-1][1] >= level:
        stack.pop()

    parent_id = stack[-1][0] if stack else page_id
    hierarchy = ["doc", page_id]
    hierarchy.extend(item[0] for item in stack)
    hierarchy.append(block_id)

    blocks.append(
        ContentBlock(
            block_id=block_id,
            block_type=BlockType.SECTION,
            text=text,
            parent_block_id=parent_id,
            order=counters["section"],
            hierarchy_path=hierarchy,
            location=location,
            metadata={
                "heading_level": level,
                "heading_text": text,
                "element_type": element_type,
                **({"category": meta["category"]} if "category" in meta else {}),
            },
        )
    )
    stack.append((block_id, level))


def _ensure_page_block(
    blocks: list[ContentBlock],
    page_blocks: dict[int, str],
    page_number: int,
) -> str:
    if page_number in page_blocks:
        return page_blocks[page_number]

    block_id = f"page_{page_number}"
    page_blocks[page_number] = block_id
    blocks.append(
        ContentBlock(
            block_id=block_id,
            block_type=BlockType.PAGE,
            text="",
            parent_block_id=None,
            order=page_number,
            hierarchy_path=["doc", block_id],
            location=BlockLocation(page_number=page_number),
            metadata={"extraction_method": "unstructured"},
        )
    )
    return block_id


def _current_parent(
    page_id: str, page_number: int, section_stack: dict[int, list[tuple[str, int]]]
) -> tuple[str, list[str]]:
    stack = section_stack.get(page_number, [])
    if not stack:
        return page_id, ["doc", page_id]
    hierarchy = ["doc", page_id, *[item[0] for item in stack]]
    return stack[-1][0], hierarchy


def _page_number(meta: dict[str, Any]) -> int:
    value = meta.get("page_number", 1)
    return int(value) if value is not None else 1


def _block_location(meta: dict[str, Any]) -> BlockLocation | None:
    page_number = meta.get("page_number")
    coordinates = meta.get("coordinates") or {}
    points = coordinates.get("points")
    bbox = _points_to_bbox(points) if points else None
    if page_number is None and bbox is None:
        return None
    return BlockLocation(
        page_number=int(page_number) if page_number is not None else None,
        bbox=bbox,
    )


def _points_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _element_metadata(element_type: str, meta: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"element_type": element_type}
    if "text_as_html" in meta:
        payload["text_as_html"] = meta["text_as_html"]
    if "emphasized_text_contents" in meta:
        payload["emphasized_text_contents"] = meta["emphasized_text_contents"]
    return payload
