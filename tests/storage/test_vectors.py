"""Tests for float32 vector helpers."""

from __future__ import annotations

import pytest

from custom_rag.storage.vectors import cosine_similarity, pack_vector, unpack_vector


def test_pack_unpack_roundtrip() -> None:
    values = [0.1, -0.2, 0.3, 0.0]
    blob = pack_vector(values)
    unpacked = unpack_vector(blob)
    assert len(unpacked) == len(values)
    assert unpacked == [pytest.approx(v, abs=1e-6) for v in values]


def test_cosine_similarity_orders_related_vectors() -> None:
    query = [1.0, 0.0, 0.0]
    close = [0.9, 0.1, 0.0]
    far = [0.0, 1.0, 0.0]
    assert cosine_similarity(query, close) > cosine_similarity(query, far)
