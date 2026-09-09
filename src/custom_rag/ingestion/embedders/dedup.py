"""Document-level dedup state for embedding skip decisions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DedupState(Protocol):
    def get_content_hash(self, source_path: str) -> str | None: ...

    def set_content_hash(self, source_path: str, content_hash: str) -> None: ...


class InMemoryDedupState:
    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def get_content_hash(self, source_path: str) -> str | None:
        return self._hashes.get(source_path)

    def set_content_hash(self, source_path: str, content_hash: str) -> None:
        self._hashes[source_path] = content_hash


def should_skip_embedding(
    *,
    source_path: str,
    content_hash: str,
    dedup_state: DedupState | None,
) -> bool:
    if dedup_state is None:
        return False
    stored = dedup_state.get_content_hash(source_path)
    return stored is not None and stored == content_hash
