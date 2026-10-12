"""Dense-vector validation and the SQLite vec0 binary representation."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from memorysum.core.errors import ProviderError


def normalize_vector(vector: Sequence[float]) -> list[float]:
    if not vector:
        raise ProviderError(
            "model_provider_invalid_embedding",
            "The embedding provider returned an empty vector.",
            status_code=503,
        )
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ProviderError(
            "model_provider_invalid_embedding",
            "The embedding provider returned a non-finite vector.",
            status_code=503,
        )
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ProviderError(
            "model_provider_invalid_embedding",
            "The embedding provider returned a zero-length vector.",
            status_code=503,
        )
    return [value / norm for value in values]


def normalize_batch(
    vectors: Sequence[Sequence[float]], expected: int
) -> tuple[list[list[float]], int]:
    if len(vectors) != expected or expected <= 0:
        raise ProviderError(
            "model_provider_invalid_embedding",
            "The embedding provider returned an unexpected number of vectors.",
            status_code=503,
        )
    dimension = len(vectors[0])
    normalized: list[list[float]] = []
    for vector in vectors:
        if len(vector) != dimension:
            raise ProviderError(
                "model_provider_invalid_embedding",
                "The embedding provider returned inconsistent dimensions.",
                status_code=503,
            )
        normalized.append(normalize_vector(vector))
    return normalized, dimension


def vector_to_blob(vector: Sequence[float]) -> bytes:
    if not vector:
        raise ValueError("vector must not be empty")
    return struct.pack("<" + "f" * len(vector), *vector)


def blob_to_vector(blob: bytes, dimension: int) -> list[float]:
    if dimension <= 0 or len(blob) != dimension * 4:
        raise ValueError("stored vector has an invalid dimension")
    return list(struct.unpack("<" + "f" * dimension, blob))
