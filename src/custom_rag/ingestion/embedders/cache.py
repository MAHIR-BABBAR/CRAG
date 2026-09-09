"""In-memory embedding cache keyed by text hash and model."""

from __future__ import annotations

from collections import OrderedDict


class EmbeddingCache:
    def __init__(self, *, max_entries: int = 10_000) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    def get(self, text_hash: str, model: str) -> list[float] | None:
        key = (text_hash, model)
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def set(self, text_hash: str, model: str, embedding: list[float]) -> None:
        key = (text_hash, model)
        self._entries[key] = embedding
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
