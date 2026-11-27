import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.api.metrics import ApiMetrics
from src.api.schemas import (
    ChunkDetailResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    MetaResponse,
)
from src.api.security import AccessControl, ProtectedRoute
from src.api.service import DiagnosticService, IndexNotReady
from src.core.embedding import create_embedding_client
from src.core.exceptions import (
    ConfigurationError,
    CorpusError,
    IndexValidationError,
    ModelClientError,
)
from src.core.llm import create_llm_client
from src.core.logging import request_id_context
from src.core.reranker import create_reranker
from src.core.settings import Settings, load_settings
from src.retrieval.index import IndexManifest

_STATIC_DIR = Path(__file__).with_name("static")
_REQUEST_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}")


async def _service(request: Request) -> DiagnosticService:
    return request.app.state.diagnostic_service


Service = Annotated[DiagnosticService, Depends(_service)]


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = (settings if settings is not None else load_settings()).model_copy(deep=True)
    metrics = ApiMetrics() if configured.api.enable_metrics else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as stack:
            embedder = stack.enter_context(create_embedding_client(configured.embedding))
            llm = stack.enter_context(create_llm_client(configured.llm))
            service = DiagnosticService(
                configured, embedder, create_reranker(configured.reranker), llm, metrics=metrics
            )
            await run_in_threadpool(service.load_existing)
            app.state.diagnostic_service = service
            yield

    app = FastAPI(title="DiagAgent-RAG", version="2.0.1", lifespan=lifespan)
    app.state.access_control = AccessControl(configured.api)
    app.router.route_class = ProtectedRoute
    if configured.api.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=configured.api.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )

    @app.middleware("http")
    async def correlate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid4().hex
        request.state.request_id = request_id
        token = request_id_context.set(request_id)
        started = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_context.reset(token)
            if metrics is not None:
                route = request.scope.get("route")
                metrics.observe_http(
                    getattr(route, "path", "unmatched"), status, perf_counter() - started
                )

    @app.exception_handler(Exception)
    async def internal_error(request: Request, _: Exception) -> JSONResponse:
        headers = {"X-Request-ID": request.state.request_id}
        return JSONResponse(
            {"detail": "An internal server error occurred."}, status_code=500, headers=headers
        )

    if configured.api.enable_demo_ui:
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def demo() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/health", response_model=HealthResponse)
    def health(service: Service) -> HealthResponse:
        return service.health()

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
    )
    async def ready(service: Service, response: Response) -> HealthResponse:
        state = service.health()
        if state.status != "ok":
            response.status_code = 503
        return state

    if metrics is not None:

        @app.get("/metrics", response_class=Response)
        def prometheus_metrics(service: Service) -> Response:
            return Response(
                metrics.render(service.health()), media_type="text/plain; version=0.0.4"
            )

    @app.get("/v1/meta", response_model=MetaResponse)
    def meta(service: Service) -> MetaResponse:
        return service.meta()

    def _read_chunk(chunk_id: str, service: DiagnosticService) -> ChunkDetailResponse:
        try:
            chunk = service.chunk_detail(chunk_id)
        except IndexNotReady as exc:
            raise HTTPException(503, str(exc)) from exc
        if chunk is None:
            raise HTTPException(404, "The chunk is not in the active index.")
        return chunk

    @app.get("/v1/chunks", response_model=ChunkDetailResponse)
    def chunk_detail_query(
        service: Service, chunk_id: str = Query(..., min_length=1)
    ) -> ChunkDetailResponse:
        return _read_chunk(chunk_id, service)

    @app.get("/v1/chunks/{chunk_id:path}", response_model=ChunkDetailResponse)
    def chunk_detail(chunk_id: str, service: Service) -> ChunkDetailResponse:
        return _read_chunk(chunk_id, service)

    @app.post("/v1/ingest", response_model=IndexManifest)
    def ingest(service: Service) -> IndexManifest:
        try:
            return service.ingest()
        except CorpusError as exc:
            raise HTTPException(
                422, "Check the configured corpus and diagnostic rule tables."
            ) from exc
        except IndexValidationError as exc:
            raise HTTPException(503, "The index could not be published or loaded.") from exc
        except ConfigurationError as exc:
            raise HTTPException(503, "The configured model provider is unavailable.") from exc
        except ModelClientError as exc:
            raise HTTPException(502, "The embedding provider could not build the index.") from exc

    @app.post("/v1/diagnose", response_model=DiagnoseResponse, response_model_exclude_none=True)
    def diagnose(request: DiagnoseRequest, service: Service) -> DiagnoseResponse:
        if len(request.query) > configured.api.max_query_chars:
            raise HTTPException(422, "The diagnosis query exceeds the configured character limit.")
        if len(request.sensor_readings) > configured.api.max_sensor_readings:
            raise HTTPException(422, "The diagnosis request exceeds the configured sensor limit.")
        try:
            return service.diagnose(request)
        except IndexNotReady as exc:
            raise HTTPException(503, str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(503, "The configured model provider is unavailable.") from exc

    return app
