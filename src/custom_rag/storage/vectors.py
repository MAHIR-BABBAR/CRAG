"""Float32 vector packing and cosine similarity."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def pack_vector(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    if len(blob) % 4 != 0:
        raise ValueError("vector blob length must be a multiple of 4")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    if not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def batch_cosine_topk(
    query: list[float],
    matrix: NDArray[np.float32],
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return ``(row_index, score)`` for the top-k rows of a float32 matrix.

    ``matrix`` is an ``(n, d)`` numpy ndarray. This scores every row, exactly as
    the pure-Python path does; it is the same search expressed as one matrix
    product rather than an approximate index.
    """
    import numpy as np

    if top_k <= 0 or matrix.shape[0] == 0:
        return []
    q = np.asarray(query, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    # matrix rows assumed L2-normalized at upsert time is not guaranteed;
    # normalize per query via row norms.
    row_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ q
    scores = np.zeros(matrix.shape[0], dtype=np.float32)
    nonzero = row_norms > 0
    scores[nonzero] = dots[nonzero] / (row_norms[nonzero] * q_norm)
    k = min(top_k, matrix.shape[0])
    if k == matrix.shape[0]:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, kth=k - 1)[:k]
        order = part[np.argsort(-scores[part])]
    return [(int(i), float(scores[i])) for i in order]
