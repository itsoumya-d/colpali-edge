"""MaxSim late-interaction scoring (ColPali, arXiv:2407.01449).

The score is the sum over query tokens of the best dot product against any
document token:

    LI(q, d) = sum_j max_i <E_q[j], E_d[i]>

No OCR, chunking, or pooling happens here. Inputs are already patch/token
embeddings projected into the same dimension D.
"""

from __future__ import annotations

from math import fsum
from typing import Sequence

Vector = Sequence[float]
Embedding = Sequence[Vector]


def _validate(embedding: Embedding, label: str, dimension: int | None = None) -> int:
    if not embedding:
        raise ValueError(f"{label} must contain at least one vector")
    width = len(embedding[0])
    if width == 0:
        raise ValueError(f"{label} vectors must have at least one dimension")
    for index, vector in enumerate(embedding):
        if len(vector) != width:
            raise ValueError(
                f"{label}[{index}] has dimension {len(vector)}, expected {width}"
            )
    if dimension is not None and width != dimension:
        raise ValueError(
            f"{label} dimension {width} does not match query dimension {dimension}"
        )
    return width


def _dot(left: Vector, right: Vector) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def maxsim(query: Embedding, document: Embedding) -> float:
    """Return the late-interaction MaxSim score for one query/document pair."""
    dimension = _validate(query, "query")
    _validate(document, "document", dimension)

    best_per_query = []
    for query_vector in query:
        best = float("-inf")
        for document_vector in document:
            score = _dot(query_vector, document_vector)
            if score > best:
                best = score
        best_per_query.append(best)
    return fsum(best_per_query)


def maxsim_batch(query: Embedding, documents: Sequence[Embedding]) -> list[float]:
    """Score one query against many documents."""
    if not documents:
        raise ValueError("documents must not be empty")
    return [maxsim(query, document) for document in documents]
