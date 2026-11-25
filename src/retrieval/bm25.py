import logging
from collections import defaultdict
from collections.abc import Sequence
from time import perf_counter

from rank_bm25 import BM25Okapi

from src.core.logging import log_event
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import SearchHit
from src.retrieval.tokenizer import tokenize_bm25

_LOGGER = logging.getLogger("diagagent.retrieval.bm25")


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = tuple(chunks)
        terms = [tokenize_bm25(chunk.retrieval_text, include_components=True) for chunk in chunks]
        self._model = BM25Okapi(terms) if any(terms) else None
        self._positions: dict[str, set[int]] = defaultdict(set)
        for index, document_terms in enumerate(terms):
            for term in set(document_terms):
                self._positions[term].add(index)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = tokenize_bm25(query)
        if not terms or self._model is None:
            return []

        started = perf_counter()
        matched = sorted({index for term in terms for index in self._positions.get(term, ())})
        scores = self._model.get_batch_scores(terms, matched)
        ranked = sorted(zip(matched, scores, strict=True), key=lambda pair: (-pair[1], pair[0]))
        hits = [
            SearchHit(chunk=self._chunks[index], score=float(score), bm25_score=float(score))
            for index, score in ranked[:top_k]
        ]
        log_event(
            _LOGGER,
            "retrieval.bm25",
            latency_ms=(perf_counter() - started) * 1000,
            top_k=top_k,
            scores=[hit.score for hit in hits],
        )
        return hits
