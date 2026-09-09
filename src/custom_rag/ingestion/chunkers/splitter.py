"""Recursive character splitter for oversized leaf blocks."""

from __future__ import annotations

_DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "; ", " ")


def split_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int = 0,
    separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    raw_parts = _recursive_split(text, max_chars, separators)
    merged = _merge_small_parts(raw_parts, min_chars=min_chars, max_chars=max_chars)
    if overlap_chars <= 0 or len(merged) <= 1:
        return merged
    return _apply_overlap(merged, overlap_chars=overlap_chars, max_chars=max_chars)


def _recursive_split(text: str, max_chars: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    for separator in separators:
        if separator and separator in text:
            parts = text.split(separator)
            chunks: list[str] = []
            current = ""
            for index, part in enumerate(parts):
                piece = part if index == len(parts) - 1 else part + separator
                candidate = piece if not current else current + piece
                if len(candidate) <= max_chars:
                    current = candidate
                    continue
                if current:
                    chunks.append(current)
                if len(piece) <= max_chars:
                    current = piece
                else:
                    if separator == " ":
                        chunks.extend(_hard_split(piece, max_chars))
                        current = ""
                    else:
                        chunks.extend(_recursive_split(piece, max_chars, separators[1:]))
                        current = ""
            if current:
                chunks.append(current)
            return [chunk for chunk in chunks if chunk]

    return _hard_split(text, max_chars)


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _merge_small_parts(parts: list[str], *, min_chars: int, max_chars: int) -> list[str]:
    if min_chars <= 0 or not parts:
        return parts

    merged: list[str] = []
    buffer = ""
    for part in parts:
        if not buffer:
            buffer = part
            continue
        if len(buffer) < min_chars and len(buffer) + len(part) <= max_chars:
            buffer += part
        else:
            merged.append(buffer)
            buffer = part
    if buffer:
        if merged and len(buffer) < min_chars and len(merged[-1]) + len(buffer) <= max_chars:
            merged[-1] += buffer
        elif len(buffer) >= min_chars or not merged:
            merged.append(buffer)
    return merged


def _apply_overlap(parts: list[str], *, overlap_chars: int, max_chars: int) -> list[str]:
    if len(parts) <= 1:
        return parts

    overlapped = [parts[0]]
    for part in parts[1:]:
        previous = overlapped[-1]
        prefix = previous[-overlap_chars:] if overlap_chars else ""
        combined = prefix + part
        if len(combined) > max_chars and prefix:
            combined = part
        overlapped.append(combined)
    return overlapped
