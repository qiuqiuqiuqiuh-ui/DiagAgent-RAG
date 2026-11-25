import logging
from time import perf_counter

from src.core.embedding import BaseEmbeddingClient
from src.core.exceptions import IndexValidationError
from src.core.logging import log_event
from src.retrieval.index import StoredIndex
from src.retrieval.schemas import SearchHit

_LOGGER = logging.getLogger("diagagent.retrieval.dense")


class DenseRetriever:
    def __init__(self, index: StoredIndex, embedder: BaseEmbeddingClient) -> None:
        manifest = index.manifest
        if (embedder.provider, embedder.model, embedder.dimension) != (
            manifest.embedding_provider,
            manifest.embedding_model,
            manifest.embedding_dim,
        ):
            raise IndexValidationError(
                "Query embedding client provider, model, and dimension must match the index"
            )
        self._chunks = index.chunks
        self._vectors = index.vectors
        self._embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self._chunks or not query.strip():
            return []

        started = perf_counter()
        embedding = self._embedder.embed([query])
        if not embedding.any():
            return []
        scores, indices = self._vectors.search(embedding, min(top_k, len(self._chunks)))
        hits = [
            SearchHit(chunk=self._chunks[index], score=float(score), dense_score=float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]
        log_event(
            _LOGGER,
            "retrieval.dense",
            latency_ms=(perf_counter() - started) * 1000,
            top_k=top_k,
            scores=[hit.score for hit in hits],
        )
        return hits
