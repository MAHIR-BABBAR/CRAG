"""Deterministic mock embedding provider for tests and CI."""

from __future__ import annotations

import hashlib
import math
import struct


class MockEmbeddingProvider:
    name = "mock"
    dimensions = 8

    def __init__(self, *, model: str = "mock-embed-v1", dimensions: int = 8) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vector(text, self.dimensions) for text in texts]


def _hash_to_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimensions):
        start = (index * 4) % len(digest)
        chunk = digest[start : start + 4]
        if len(chunk) < 4:
            chunk = chunk + digest[: 4 - len(chunk)]
        raw = struct.unpack("!I", chunk)[0]
        values.append((raw / 2**32) * 2 - 1)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
