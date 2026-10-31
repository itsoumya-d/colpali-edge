"""ColPali-style late-interaction primitives.

This package is the deterministic algorithm core. It does not download model
weights, run a vision encoder, or claim a benchmark result. The model adapters
and GPU training live outside this dependency-free package.
"""

from .maxsim import maxsim, maxsim_batch
from .loss import (
    ContrastiveLossResult,
    contrastive_loss,
    in_batch_contrastive_loss,
)
from .pooling import hierarchical_pool
from .quantize import binary_quantize, binary_hamming, binary_dot_approx
from .index import MultiVectorIndex, SearchHit
from .encoder import (
    ColPaliEncoder,
    EncoderUnavailableError,
    SyntheticEncoder,
    TransformersColPaliEncoder,
)
from .heatmap import AttentionMap, heatmap_payload, maxsim_attention

__all__ = [
    "maxsim",
    "maxsim_batch",
    "ContrastiveLossResult",
    "contrastive_loss",
    "in_batch_contrastive_loss",
    "hierarchical_pool",
    "binary_quantize",
    "binary_hamming",
    "binary_dot_approx",
    "MultiVectorIndex",
    "SearchHit",
    "ColPaliEncoder",
    "EncoderUnavailableError",
    "SyntheticEncoder",
    "TransformersColPaliEncoder",
    "AttentionMap",
    "heatmap_payload",
    "maxsim_attention",
]
