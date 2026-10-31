"""Two-stage multi-vector index: binary coarse scan + FP32 MaxSim re-rank.

The index keeps both scores on every hit. A re-ranking result must not overwrite
the coarse score, otherwise an ablation cannot attribute ranking changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .maxsim import Embedding, maxsim
from .quantize import binary_dot_approx, binary_quantize


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    rank: int
    score: float
    coarse_score: int


class MultiVectorIndex:
    """In-memory index for ColPali-style multi-vector documents."""

    def __init__(self) -> None:
        self._documents: dict[str, Embedding] = {}
        self._binary: dict[str, tuple[bytes, ...]] = {}
        self._dimension: int | None = None

    def add(self, doc_id: str, embeddings: Embedding) -> None:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if doc_id in self._documents:
            raise ValueError(f"document already indexed: {doc_id}")
        if not embeddings:
            raise ValueError("embeddings must not be empty")

        width = len(embeddings[0])
        if width == 0:
            raise ValueError("embedding vectors must have at least one dimension")
        if self._dimension is None:
            self._dimension = width
        elif width != self._dimension:
            raise ValueError(
                f"document dimension {width} does not match index dimension {self._dimension}"
            )
        for index, vector in enumerate(embeddings):
            if len(vector) != width:
                raise ValueError(
                    f"embeddings[{index}] has dimension {len(vector)}, expected {width}"
                )

        self._documents[doc_id] = tuple(tuple(vector) for vector in embeddings)
        self._binary[doc_id] = tuple(binary_quantize(vector) for vector in embeddings)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def _coarse_score(self, query: Embedding, doc_id: str) -> int:
        query_bits = [binary_quantize(vector) for vector in query]
        document_bits = self._binary[doc_id]
        total = 0
        for q_bits in query_bits:
            best = max(
                binary_dot_approx(q_bits, d_bits, self._dimension or 0)
                for d_bits in document_bits
            )
            total += best
        return total

    def _validate_query(self, query: Embedding) -> None:
        if not self._documents:
            raise ValueError("index is empty")
        if not query:
            raise ValueError("query must not be empty")
        width = len(query[0])
        if width == 0 or any(len(vector) != width for vector in query):
            raise ValueError("query vectors must share a non-zero dimension")
        if self._dimension is not None and width != self._dimension:
            raise ValueError(
                f"query dimension {width} does not match index dimension {self._dimension}"
            )

    def coarse_scores(self, query: Embedding) -> list[tuple[str, int]]:
        """All binary coarse scores, descending. Exposed for ablations."""
        self._validate_query(query)
        coarse = [
            (doc_id, self._coarse_score(query, doc_id))
            for doc_id in self._documents
        ]
        coarse.sort(key=lambda item: (-item[1], item[0]))
        return coarse

    def search_coarse(self, query: Embedding, top_k: int = 5) -> list[SearchHit]:
        """Binary-only search. The hit score is the coarse integer score."""
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        coarse = self.coarse_scores(query)[:top_k]
        return [
            SearchHit(
                doc_id=doc_id,
                rank=rank,
                score=float(score),
                coarse_score=score,
            )
            for rank, (doc_id, score) in enumerate(coarse, start=1)
        ]

    def search(
        self,
        query: Embedding,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[SearchHit]:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(candidate_k, int) or candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")

        coarse = self.coarse_scores(query)
        candidates = coarse[:min(candidate_k, len(coarse))]

        refined = [
            (doc_id, coarse_score, maxsim(query, self._documents[doc_id]))
            for doc_id, coarse_score in candidates
        ]
        refined.sort(key=lambda item: (-item[2], item[0]))

        return [
            SearchHit(
                doc_id=doc_id,
                rank=rank,
                score=score,
                coarse_score=coarse_score,
            )
            for rank, (doc_id, coarse_score, score) in enumerate(refined[:top_k], start=1)
        ]
