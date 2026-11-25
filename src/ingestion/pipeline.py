import logging
from time import perf_counter

import faiss

from src.core.embedding import BaseEmbeddingClient
from src.core.exceptions import ConfigurationError
from src.core.logging import log_event
from src.core.settings import Settings
from src.ingestion.corpus import read_corpus
from src.retrieval.index import IndexManifest, save_index

_LOGGER = logging.getLogger("diagagent.ingestion")


def build_index(settings: Settings, embedder: BaseEmbeddingClient) -> IndexManifest:
    if (embedder.provider, embedder.model, embedder.dimension) != (
        settings.embedding.provider,
        settings.embedding.model,
        settings.embedding.dimension,
    ):
        raise ConfigurationError(
            "Embedding client provider, model, and dimension must match the indexing profile"
        )

    started = perf_counter()
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    index = faiss.IndexFlatIP(embedder.dimension)
    batch_size = settings.embedding.batch_size
    for offset in range(0, len(corpus.chunks), batch_size):
        batch = corpus.chunks[offset : offset + batch_size]
        index.add(embedder.embed([chunk.retrieval_text for chunk in batch]))
    manifest = save_index(
        settings.paths.index_dir,
        index,
        corpus.chunks,
        corpus.corpus_hash,
        settings.embedding,
        settings.ingestion,
    )
    log_event(
        _LOGGER,
        "index.built",
        index_version=manifest.index_version,
        document_count=corpus.document_count,
        chunk_count=manifest.chunk_count,
        latency_ms=(perf_counter() - started) * 1000,
    )
    return manifest
