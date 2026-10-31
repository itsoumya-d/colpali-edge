"""Encoder interfaces for ColPali-style multi-vector retrieval.

The deterministic ``SyntheticEncoder`` is a test fixture. The Transformers
adapter is provided for environments with model weights and a GPU/CPU runtime;
it is not exercised in this repository's CI and makes no benchmark claim.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

from .maxsim import Embedding


class EncoderUnavailableError(RuntimeError):
    """Raised when an adapter's runtime or weights are unavailable."""


class ColPaliEncoder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode_query(self, text: str) -> Embedding: ...

    def encode_page(self, image: Any) -> Embedding: ...


class SyntheticEncoder:
    """Deterministic hashed embeddings for tests and UI fixtures.

    This is NOT a vision-language model. It hashes the input, expands the digest
    into float features, and returns one vector per synthetic token so the
    MaxSim/index plumbing can be tested without weights.
    """

    def __init__(self, dimension: int = 32, tokens: int = 8, seed: str = "colpali-edge") -> None:
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        if not isinstance(seed, str) or not seed:
            raise ValueError("seed must be a non-empty string")
        self._dimension = dimension
        self._tokens = tokens
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vectors(self, payload: bytes) -> Embedding:
        vectors = []
        for token_index in range(self._tokens):
            digest = hashlib.sha256(
                self._seed.encode() + payload + token_index.to_bytes(4, "big")
            ).digest()
            values = []
            for dimension_index in range(self._dimension):
                byte = digest[dimension_index % len(digest)]
                values.append((byte / 255.0) * 2.0 - 1.0)
            vectors.append(values)
        return vectors

    def encode_query(self, text: str) -> Embedding:
        return self._vectors(b"query:" + text.encode("utf-8"))

    def encode_page(self, image: Any) -> Embedding:
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
        elif isinstance(image, str):
            payload = image.encode("utf-8")
        else:
            payload = repr(image).encode("utf-8")
        return self._vectors(b"page:" + payload)


class TransformersColPaliEncoder:
    """Lazy adapter for ColPali/ColQwen2 checkpoints.

    Requires ``torch`` and ``transformers`` plus downloaded weights. The adapter
    reads the model's ``embeddings`` output when present and falls back to the
    last hidden state otherwise. It is intentionally lazy so importing
    ``colpali_edge`` stays dependency-free.
    """

    def __init__(
        self,
        model_id: str = "vidore/colpali-v1.2",
        device: str | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None
        self._torch = None
        self._dimension: int | None = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EncoderUnavailableError("call load() before reading dimension")
        return self._dimension

    def load(self) -> None:
        if not self.available():
            raise EncoderUnavailableError(
                "TransformersColPaliEncoder requires torch and transformers. "
                "Install them and provide model weights; this repository's CI "
                "uses SyntheticEncoder instead."
            )
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        if self.device:
            self._model = self._model.to(self.device)
        self._model.eval()
        # The projection dimension is model-specific; discover it from config.
        config = getattr(self._model, "config", None)
        dimension = getattr(config, "hidden_size", None) or getattr(config, "dim", None)
        self._dimension = int(dimension) if dimension else None
        if self._dimension is None:
            raise EncoderUnavailableError(
                "could not determine embedding dimension from model config"
            )

    def _encode(self, **inputs: Any) -> Embedding:
        if self._model is None or self._processor is None or self._torch is None:
            raise EncoderUnavailableError("call load() before encoding")
        prepared = self._processor(**inputs, return_tensors="pt")
        if self.device:
            prepared = {key: value.to(self.device) for key, value in prepared.items()}
        with self._torch.no_grad():
            output = self._model(**prepared)
        embeddings = getattr(output, "embeddings", None)
        if embeddings is None:
            embeddings = getattr(output, "last_hidden_state", None)
        if embeddings is None:
            raise EncoderUnavailableError(
                "model output exposes neither embeddings nor last_hidden_state"
            )
        tensor = embeddings[0] if embeddings.ndim == 3 else embeddings
        return tensor.float().cpu().tolist()

    def encode_query(self, text: str) -> Embedding:
        return self._encode(text=text)

    def encode_page(self, image: Any) -> Embedding:
        return self._encode(images=image)
