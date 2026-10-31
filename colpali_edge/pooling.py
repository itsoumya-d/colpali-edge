"""Deterministic hierarchical token pooling for multi-vector embeddings."""

from __future__ import annotations

from math import fsum
from typing import Sequence

from .maxsim import Embedding, Vector


def hierarchical_pool(embeddings: Embedding, factor: int = 2) -> list[Vector]:
    """Mean-pool consecutive vectors in groups of ``factor``.

    A short final group is averaged over its actual length, so the output
    length is ``ceil(len(embeddings) / factor)``. This reduces the number of
    vectors the MaxSim operator must score without learning new weights.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if not embeddings:
        raise ValueError("embeddings must not be empty")

    width = len(embeddings[0])
    if width == 0:
        raise ValueError("embedding vectors must have at least one dimension")
    for index, vector in enumerate(embeddings):
        if len(vector) != width:
            raise ValueError(
                f"embeddings[{index}] has dimension {len(vector)}, expected {width}"
            )

    pooled: list[Vector] = []
    for start in range(0, len(embeddings), factor):
        group = embeddings[start:start + factor]
        pooled.append([
            fsum(vector[position] for vector in group) / len(group)
            for position in range(width)
        ])
    return pooled
