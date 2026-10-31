"""MaxSim attention maps for visual grounding UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .maxsim import Embedding


@dataclass(frozen=True)
class AttentionMap:
    scores: tuple[tuple[float, ...], ...]
    best_indices: tuple[int, ...]
    best_scores: tuple[float, ...]
    normalized: tuple[tuple[float, ...], ...]


def _validate(embedding: Embedding, label: str) -> int:
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
    return width


def maxsim_attention(query: Embedding, document: Embedding) -> AttentionMap:
    """Return the full query-token × document-patch similarity matrix."""
    dimension = _validate(query, "query")
    _validate(document, "document")
    if len(document[0]) != dimension:
        raise ValueError("query and document dimensions do not match")

    scores: list[tuple[float, ...]] = []
    best_indices: list[int] = []
    best_scores: list[float] = []

    for query_vector in query:
        row = []
        best_value = float("-inf")
        best_index = 0
        for patch_index, document_vector in enumerate(document):
            value = sum(a * b for a, b in zip(query_vector, document_vector))
            row.append(value)
            if value > best_value:
                best_value = value
                best_index = patch_index
        scores.append(tuple(row))
        best_indices.append(best_index)
        best_scores.append(best_value)

    flat = [value for row in scores for value in row]
    low = min(flat)
    high = max(flat)
    span = high - low
    normalized = tuple(
        tuple(0.5 if span == 0 else (value - low) / span for value in row)
        for row in scores
    )

    return AttentionMap(
        scores=tuple(scores),
        best_indices=tuple(best_indices),
        best_scores=tuple(best_scores),
        normalized=normalized,
    )


def heatmap_payload(
    query_labels: Sequence[str],
    patch_labels: Sequence[str],
    attention: AttentionMap,
) -> dict:
    if len(query_labels) != len(attention.scores):
        raise ValueError("query_labels length must match attention rows")
    if not patch_labels or len(patch_labels) != len(attention.scores[0]):
        raise ValueError("patch_labels length must match attention columns")

    return {
        "query_labels": list(query_labels),
        "patch_labels": list(patch_labels),
        "rows": [
            {
                "query": query_labels[index],
                "scores": list(attention.scores[index]),
                "normalized": list(attention.normalized[index]),
                "best_patch": attention.best_indices[index],
                "best_score": attention.best_scores[index],
            }
            for index in range(len(query_labels))
        ],
    }
