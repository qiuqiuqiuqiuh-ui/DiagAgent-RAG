import logging
from collections.abc import Sequence
from time import perf_counter

from src.core.embedding import BaseEmbeddingClient
from src.core.logging import log_event
from src.core.reranker import BaseReranker
from src.core.settings import RetrievalSettings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.index import StoredIndex
from src.retrieval.schemas import SearchHit

_LOGGER = logging.getLogger("diagagent.retrieval.hybrid")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rank_constant: int = 60
) -> list[tuple[str, float]]:
    """Merge rankings by summing 1 / (rank_constant + rank), with one-based ranks."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


class HybridRetriever:
    def __init__(
        self,
        index: StoredIndex,
        embedder: BaseEmbeddingClient,
        reranker: BaseReranker | None = None,
        *,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.manifest = index.manifest
        self.dense = DenseRetriever(index, embedder)
        self.bm25 = BM25Retriever(index.chunks)
        self._reranker = reranker
        self._settings = settings if settings is not None else RetrievalSettings()

    def search(self, query: str, top_k: int = 5, *, rerank: bool = True) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = perf_counter()
        candidate_k = max(top_k, self._settings.candidate_k)
        dense = {hit.chunk.chunk_id: hit for hit in self.dense.search(query, candidate_k)}
        sparse = {hit.chunk.chunk_id: hit for hit in self.bm25.search(query, candidate_k)}
        candidates = dense | sparse
        scores = reciprocal_rank_fusion(
            [list(dense), list(sparse)], self._settings.rrf_rank_constant
        )
        hits = [
            SearchHit(
                chunk=candidates[chunk_id].chunk,
                score=score,
                dense_score=dense[chunk_id].dense_score if chunk_id in dense else None,
                bm25_score=sparse[chunk_id].bm25_score if chunk_id in sparse else None,
                rrf_score=score,
            )
            for chunk_id, score in scores[:candidate_k]
        ]

        if rerank and self._reranker is not None and hits:
            reranker_scores = self._reranker.score(
                query, [hit.chunk.retrieval_text for hit in hits]
            )
            hits = sorted(
                (
                    hit.model_copy(update={"score": score, "reranker_score": score})
                    for hit, score in zip(hits, reranker_scores, strict=True)
                ),
                key=lambda hit: -hit.score,
            )
        result = hits[:top_k]
        log_event(
            _LOGGER,
            "retrieval.hybrid",
            index_version=self.manifest.index_version,
            latency_ms=(perf_counter() - started) * 1000,
            top_k=top_k,
            scores=[hit.score for hit in result],
        )
        return result
