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

from src.core.client import ManagedClient
from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.logging import log_event
from src.core.openai_compatible import OpenAICompatibleTransport
from src.core.settings import EmbeddingSettings
from src.core.tokenizer import tokenize

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_LOGGER = logging.getLogger("diagagent.core.embedding")


class BaseEmbeddingClient(ManagedClient, ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider identity used when matching the document and query vector spaces."""

    def __init__(self, model: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        self.model = model
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return finite float32 embeddings, with each nonzero row L2-normalized."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not str/bytes")
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
        if vectors.shape != expected_shape:
            raise ModelClientError(
                f"Expected embeddings with shape {expected_shape}, got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ModelClientError("Expected finite embeddings")
        # Accumulate norms in float64 to handle finite float32 extremes without overflow.
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        normalized = np.divide(
            vectors, np.where(norms == 0, 1, norms), out=np.empty_like(vectors, order="C")
        )
        log_event(
            _LOGGER,
            "embedding.embed",
            model=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            input_count=len(texts),
            embedding_dim=self.dimension,
        )
        return normalized

    @abstractmethod
    def _embed(self, texts: Sequence[str]) -> ArrayLike:
        """Embed each input in its original position."""


class MockEmbeddingClient(BaseEmbeddingClient):
    provider = "mock"

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
        return vectors


class _EmbeddingItem(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float]


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingItem]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    provider = "openai_compatible"

    def __init__(
        self, settings: EmbeddingSettings, *, http_client: httpx.Client | None = None
    ) -> None:
        super().__init__(settings.model, settings.dimension)
        self._request_dimensions = settings.request_dimensions
        self._batch_size = settings.batch_size
        self._transport = OpenAICompatibleTransport(
            str(settings.base_url),
            settings.api_key,
            settings.timeout_seconds,
            max_retries=settings.max_retries,
            client=http_client,
        )

    def close(self) -> None:
        self._transport.close()

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = list(texts[offset : offset + self._batch_size])
            payload: dict[str, object] = {"model": self.model, "input": batch}
            if self._request_dimensions:
                payload["dimensions"] = self.dimension
            result = self._transport.post("embeddings", payload, _EmbeddingResponse)
            ordered = sorted(result.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ModelClientError("Embedding results must cover each input index exactly once")
            embeddings.extend(item.embedding for item in ordered)
        return embeddings


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    provider = "sentence_transformers"

    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(settings.model, settings.dimension)
        self._device = settings.device
        self._batch_size = settings.batch_size

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
                batch_size=self._batch_size,
                convert_to_numpy=True,
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
    if settings.provider == "sentence_transformers":
        return SentenceTransformerEmbeddingClient(settings)
    raise ConfigurationError("Unsupported embedding provider")
