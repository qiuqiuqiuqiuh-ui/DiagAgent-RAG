import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from hashlib import blake2b
from time import perf_counter
from typing import TYPE_CHECKING

import httpx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, Field

from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.logging import log_event
from src.core.openai_compatible import OpenAICompatibleTransport
from src.core.settings import EmbeddingSettings
from src.core.tokenizer import tokenize

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_LOGGER = logging.getLogger("diagagent.core.embedding")


class BaseEmbeddingClient(ABC):
    def __init__(self, model: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        self.model = model
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return a finite float32 matrix shaped (number of texts, dimension)."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        started = perf_counter()
        values = self._embed(texts)
        try:
            with np.errstate(over="ignore"):
                vectors = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            raise ModelClientError("The embedding provider returned invalid vectors") from None
        expected_shape = (len(texts), self.dimension)
        if vectors.shape != expected_shape or not np.isfinite(vectors).all():
            raise ModelClientError(f"Expected finite embeddings with shape {expected_shape}")
        log_event(
            _LOGGER,
            "embedding.embed",
            model=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            input_count=len(texts),
            embedding_dim=self.dimension,
        )
        return vectors

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> ArrayLike:
        """Embed each input in its original position."""


class MockEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, model: str = "mock-hash-v1", dimension: int = 384) -> None:
        super().__init__(model, dimension)
        self._hash_key = blake2b(b"42", digest_size=16).digest()

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for text, vector in zip(texts, vectors, strict=True):
            for token in tokenize(text):
                digest = blake2b(token.encode("utf-8"), key=self._hash_key, digest_size=9).digest()
                position = int.from_bytes(digest[:8], "big") % self.dimension
                vector[position] += 1.0 if digest[-1] & 1 else -1.0
            norm = np.linalg.norm(vector)
            if norm:
                vector /= norm
        return vectors


class _EmbeddingItem(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float]


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingItem]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self, settings: EmbeddingSettings, *, http_client: httpx.Client | None = None
    ) -> None:
        super().__init__(settings.model, settings.dimension)
        self._request_dimensions = settings.request_dimensions
        self._transport = OpenAICompatibleTransport(
            str(settings.base_url), settings.api_key, settings.timeout_seconds, client=http_client
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model, "input": list(texts)}
        if self._request_dimensions:
            payload["dimensions"] = self.dimension
        result = self._transport.post("embeddings", payload, _EmbeddingResponse)
        ordered = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ModelClientError("Embedding results must cover each input index exactly once")
        return [item.embedding for item in ordered]


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(settings.model, settings.dimension)
        self._device = settings.device

    @cached_property
    def _encoder(self) -> "SentenceTransformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ConfigurationError(
                "Install the huggingface extra for sentence-transformers"
            ) from None
        return SentenceTransformer(self.model, device=self._device, trust_remote_code=False)

    def _embed(self, texts: Sequence[str]) -> ArrayLike:
        try:
            return self._encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelClientError("Sentence-transformer embedding failed") from exc


def create_embedding_client(
    settings: EmbeddingSettings, *, http_client: httpx.Client | None = None
) -> BaseEmbeddingClient:
    if settings.provider == "mock":
        return MockEmbeddingClient(settings.model, settings.dimension)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(settings, http_client=http_client)
    return SentenceTransformerEmbeddingClient(settings)
