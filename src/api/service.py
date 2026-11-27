import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from types import MappingProxyType

from src.agents.tools import DiagnosticTools
from src.agents.workflow import DiagnosticWorkflow, create_diagnostic_workflow
from src.api.cache import CacheScope, SQLiteSemanticCache, context_hash, profile_hash
from src.api.schemas import (
    ChunkDetailResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    DomainMetadata,
    HealthResponse,
    MetaResponse,
)
from src.core.embedding import BaseEmbeddingClient
from src.core.exceptions import CorpusError, DiagAgentError, IndexValidationError, ModelClientError
from src.core.llm import BaseLLMClient
from src.core.logging import log_event
from src.core.reranker import BaseReranker
from src.core.settings import Settings
from src.ingestion.corpus import read_corpus
from src.ingestion.pipeline import build_index
from src.ingestion.schemas import Chunk
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.index import IndexManifest, load_index

_LOGGER = logging.getLogger("diagagent.api")


class IndexNotReady(DiagAgentError):
    """The service needs a valid index before running a diagnosis."""


@dataclass(frozen=True)
class _Snapshot:
    manifest: IndexManifest
    tools: DiagnosticTools
    workflow: DiagnosticWorkflow
    chunks_by_id: Mapping[str, Chunk]


class DiagnosticService:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbeddingClient,
        reranker: BaseReranker,
        llm: BaseLLMClient,
    ) -> None:
        self._settings = settings
        self._embedding = embedder
        self._reranker = reranker
        self._llm = llm
        self._cache = SQLiteSemanticCache(
            settings.paths.cache_file, settings.cache.similarity_threshold
        )
        self._profile_hash = profile_hash(settings)
        self._snapshot: _Snapshot | None = None
        self._ingest_lock = Lock()
        self._cache_write_lock = Lock()

    def _load_snapshot(self, *, rebuild: bool) -> _Snapshot:
        settings = self._settings
        corpus = read_corpus(
            settings.paths.corpus_dir, settings.ingestion, capture=settings.rules.source_paths
        )
        tools = DiagnosticTools.from_sources(corpus.captured_sources, settings.rules)
        if rebuild:
            build_index(settings, self._embedding, corpus=corpus)
        stored = load_index(
            settings.paths.index_dir,
            settings.embedding,
            settings.ingestion,
            expected_corpus_hash=corpus.corpus_hash,
        )
        retriever = HybridRetriever(
            stored, self._embedding, self._reranker, settings=settings.retrieval
        )
        workflow = create_diagnostic_workflow(
            settings, retriever, llm_client=self._llm, tools=tools
        )
        chunks_by_id = MappingProxyType({chunk.chunk_id: chunk for chunk in stored.chunks})
        return _Snapshot(stored.manifest, tools, workflow, chunks_by_id)

    def load_existing(self) -> None:
        if not (self._settings.paths.index_dir / "manifest.json").exists():
            return
        try:
            self._snapshot = self._load_snapshot(rebuild=False)
        except (CorpusError, IndexValidationError) as exc:
            log_event(
                _LOGGER, "api.index_load_failed", level=logging.WARNING, reason=type(exc).__name__
            )

    def health(self) -> HealthResponse:
        snapshot = self._snapshot
        if snapshot is None:
            return HealthResponse(status="not_ready")
        return HealthResponse(
            status="ok",
            index_version=snapshot.manifest.index_version,
            chunk_count=snapshot.manifest.chunk_count,
        )

    def meta(self) -> MetaResponse:
        settings = self._settings
        snapshot = self._snapshot
        tools = snapshot.tools if snapshot is not None else None
        offline = all(
            profile.provider == "mock"
            for profile in (settings.llm, settings.embedding, settings.reranker)
        )
        return MetaResponse(
            domain=DomainMetadata(name=settings.domain.name, language=settings.domain.language),
            mode="mock" if offline else "live",
            llm_model=settings.llm.model,
            embedding_model=settings.embedding.model,
            reranker_model=settings.reranker.model,
            equipment_ids=tools.equipment_ids if tools is not None else (),
            fault_codes=tools.fault_codes_by_equipment if tools is not None else {},
            sensors=tools.sensor_rules if tools is not None else (),
        )

    def chunk_detail(self, chunk_id: str) -> ChunkDetailResponse | None:
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady(
                "Build a compatible index with POST /v1/ingest before viewing evidence."
            )
        chunk = snapshot.chunks_by_id.get(chunk_id)
        return ChunkDetailResponse.model_validate(chunk, from_attributes=True) if chunk else None

    def ingest(self) -> IndexManifest:
        with self._ingest_lock:
            snapshot = self._load_snapshot(rebuild=True)
            with self._cache_write_lock:
                self._snapshot = snapshot
                try:
                    self._cache.prune_other_versions(snapshot.manifest.index_version)
                except sqlite3.Error as exc:
                    log_event(
                        _LOGGER,
                        "cache.prune_failed",
                        level=logging.WARNING,
                        reason=type(exc).__name__,
                    )
            return snapshot.manifest

    def diagnose(self, request: DiagnoseRequest) -> DiagnoseResponse:
        started = perf_counter()
        snapshot = self._snapshot
        if snapshot is None:
            raise IndexNotReady("Build a compatible index with POST /v1/ingest before diagnosing.")

        diagnosis = request.diagnostic_request()
        scope = embedding = result = None
        if not request.bypass_cache:
            scope = CacheScope(
                snapshot.manifest.index_version,
                self._profile_hash,
                context_hash(diagnosis, snapshot.tools),
            )
            try:
                embedding = self._embedding.embed([diagnosis.query])[0]
                result = self._cache.lookup(scope, embedding)
            except (sqlite3.Error, ModelClientError) as exc:
                log_event(
                    _LOGGER, "cache.lookup_failed", level=logging.WARNING, reason=type(exc).__name__
                )

        cache_hit = result is not None
        if result is None:
            result = snapshot.workflow.invoke(diagnosis, include_trace=request.include_trace)
            if scope is not None and embedding is not None:
                try:
                    with self._cache_write_lock:
                        active = self._snapshot
                        # A diagnosis that used the previous snapshot may finish after a rebuild.
                        if (
                            active is not None
                            and active.manifest.index_version == scope.index_version
                        ):
                            self._cache.store(scope, diagnosis.query, embedding, result)
                except sqlite3.Error as exc:
                    log_event(
                        _LOGGER,
                        "cache.store_failed",
                        level=logging.WARNING,
                        reason=type(exc).__name__,
                    )

        log_event(
            _LOGGER,
            "api.diagnose",
            cache_hit=cache_hit,
            bypass_cache=request.bypass_cache,
            index_version=snapshot.manifest.index_version,
            status=result.status,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return DiagnoseResponse(**result.model_dump(), cache_hit=cache_hit)
