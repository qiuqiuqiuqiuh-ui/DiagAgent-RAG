import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from src.core.exceptions import ConfigurationError, ModelClientError
from src.core.logging import log_event
from src.core.settings import RerankerSettings
from src.core.tokenizer import code_fallback_terms, tokenize_codes

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_LOGGER = logging.getLogger("diagagent.core.reranker")


class BaseReranker(ABC):
    def __init__(self, model: str) -> None:
        self.model = model

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        """Return one finite relevance score per candidate in input order."""
        if isinstance(candidates, (str, bytes)):
            raise TypeError("candidates must be a sequence of strings, not str/bytes")
        if not candidates:
            return []

        started = perf_counter()
        values = self._score(query, candidates)
        try:
            scores = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            raise ModelClientError("The reranker returned invalid scores") from None
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ModelClientError("Expected one finite reranker score for each candidate")
        results = scores.tolist()
        log_event(
            _LOGGER,
            "reranker.score",
            model=self.model,
            latency_ms=(perf_counter() - started) * 1000,
            candidate_count=len(candidates),
            scores=results,
        )
        return results

    @abstractmethod
    def _score(self, query: str, candidates: Sequence[str]) -> ArrayLike:
        """Score query and candidate pairs."""


class MockReranker(BaseReranker):
    def __init__(self, model: str = "mock-keyword-v1") -> None:
        super().__init__(model)

    def _score(self, query: str, candidates: Sequence[str]) -> list[float]:
        terms = set(tokenize_codes(query))
        if not terms:
            return [0.0] * len(candidates)
        scores = []
        for candidate in candidates:
            candidate_terms = set(tokenize_codes(candidate, include_components=True))
            exact = terms & candidate_terms
            partial = sum(
                not candidate_terms.isdisjoint(code_fallback_terms(term)) for term in terms - exact
            )
            scores.append((len(exact) + 0.5 * partial) / len(terms))
        return scores


class CrossEncoderReranker(BaseReranker):
    def __init__(self, settings: RerankerSettings) -> None:
        super().__init__(settings.model)
        self._device = settings.device
        self._inference_lock = Lock()

    @cached_property
    def _encoder(self) -> "CrossEncoder":
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ConfigurationError(
                "Install the huggingface extra for sentence-transformers"
            ) from None
        return CrossEncoder(self.model, device=self._device, trust_remote_code=False)

    def _score(self, query: str, candidates: Sequence[str]) -> ArrayLike:
        try:
            with self._inference_lock:
                result = self._encoder.predict(
                    [(query, text) for text in candidates],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            scores = np.asarray(result)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelClientError("Cross-encoder scoring failed") from exc
        return scores[:, 0] if scores.shape == (len(candidates), 1) else scores


def create_reranker(settings: RerankerSettings) -> BaseReranker:
    if settings.provider == "mock":
        return MockReranker(settings.model)
    if settings.provider == "sentence_transformers":
        return CrossEncoderReranker(settings)
    raise ConfigurationError("Unsupported reranker provider")
