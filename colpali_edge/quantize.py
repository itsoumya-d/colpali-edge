"""Binary quantization for multi-vector retrieval.

Each float vector is reduced to one sign bit per dimension, packed LSB-first
into bytes. Hamming distance gives a coarse approximation of the dot product:

    dot(a, b) ~= D - 2 * hamming(bits(a), bits(b))

This is the same compression idea used by Qdrant binary quantization; it is a
coarse stage, not a replacement for FP32 MaxSim re-ranking.
"""

from __future__ import annotations

from math import isfinite
from typing import Sequence

Vector = Sequence[float]


def binary_quantize(vector: Vector, threshold: float = 0.0) -> bytes:
    """Pack one sign bit per dimension into bytes (LSB-first)."""
    if not vector:
        raise ValueError("vector must not be empty")
    if not isfinite(threshold):
        raise ValueError("threshold must be finite")

    packed = bytearray((len(vector) + 7) // 8)
    for index, value in enumerate(vector):
        if not isfinite(value):
            raise ValueError(f"vector[{index}] must be finite")
        if value > threshold:
            packed[index >> 3] |= 1 << (index & 7)
    return bytes(packed)


def _validate_bits(left: bytes, right: bytes) -> None:
    if not isinstance(left, bytes) or not isinstance(right, bytes):
        raise TypeError("binary vectors must be bytes")
    if len(left) != len(right):
        raise ValueError("binary vectors must have the same byte length")


def binary_hamming(left: bytes, right: bytes) -> int:
    """Number of differing bits between two packed vectors."""
    _validate_bits(left, right)
    total = 0
    for a, b in zip(left, right):
        total += (a ^ b).bit_count()
    return total


def binary_dot_approx(left: bytes, right: bytes, dimension: int) -> int:
    """Approximate the dot product of two sign vectors."""
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    _validate_bits(left, right)
    if dimension > len(left) * 8:
        raise ValueError("dimension exceeds the packed vector capacity")
    return dimension - 2 * binary_hamming(left, right)
