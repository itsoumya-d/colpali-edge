"""In-batch contrastive loss used to train ColPali-style retrievers.

For a batch of b aligned (query, positive document) pairs:

    s_k+ = LI(q_k, d_k)
    s_k- = max_{l != k} LI(q_k, d_l)
    L    = -1/b * sum_k log(exp(s_k+) / (exp(s_k+) + exp(s_k-)))

The loss is computed in a numerically stable softplus form. A batch of one has
no negative and is rejected instead of returning a meaningless zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, log1p
from typing import Sequence

from .maxsim import Embedding, maxsim


@dataclass(frozen=True)
class ContrastiveLossResult:
    loss: float
    positives: tuple[float, ...]
    negatives: tuple[float, ...]


def _softplus(value: float) -> float:
    # log(1 + exp(value)) without overflow.
    if value > 0:
        return value + log1p(exp(-value))
    return log1p(exp(value))


def contrastive_loss(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float:
    """Mean softplus(negative - positive) over a batch."""
    if len(positives) != len(negatives):
        raise ValueError("positives and negatives must have the same length")
    if not positives:
        raise ValueError("loss requires at least one example")
    terms = [_softplus(negative - positive) for positive, negative in zip(positives, negatives)]
    return fsum(terms) / len(terms)


def in_batch_contrastive_loss(
    queries: Sequence[Embedding],
    documents: Sequence[Embedding],
) -> ContrastiveLossResult:
    """Compute the in-batch contrastive loss for aligned query/document pairs."""
    if len(queries) != len(documents):
        raise ValueError("queries and documents must have the same batch length")
    if len(queries) < 2:
        raise ValueError("in-batch contrastive loss requires at least two examples")

    positives: list[float] = []
    negatives: list[float] = []

    for index, query in enumerate(queries):
        scores = [maxsim(query, document) for document in documents]
        positive = scores[index]
        negative = max(
            score for position, score in enumerate(scores) if position != index
        )
        positives.append(positive)
        negatives.append(negative)

    return ContrastiveLossResult(
        loss=contrastive_loss(positives, negatives),
        positives=tuple(positives),
        negatives=tuple(negatives),
    )
