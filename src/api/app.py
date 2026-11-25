from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from src.api.schemas import DiagnoseRequest, DiagnoseResponse, HealthResponse
from src.api.service import DiagnosticService, IndexNotReady
from src.core.embedding import create_embedding_client
from src.core.exceptions import (
    ConfigurationError,
    CorpusError,
    IndexValidationError,
    ModelClientError,
)
from src.core.llm import create_llm_client
from src.core.reranker import create_reranker
from src.core.settings import Settings, load_settings
from src.retrieval.index import IndexManifest


def _service(request: Request) -> DiagnosticService:
    return request.app.state.diagnostic_service


Service = Annotated[DiagnosticService, Depends(_service)]


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = (settings if settings is not None else load_settings()).model_copy(deep=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with ExitStack() as stack:
            embedder = stack.enter_context(create_embedding_client(configured.embedding))
            llm = stack.enter_context(create_llm_client(configured.llm))
            service = DiagnosticService(
                configured, embedder, create_reranker(configured.reranker), llm
            )
            await run_in_threadpool(service.load_existing)
            app.state.diagnostic_service = service
            yield

    app = FastAPI(title="DiagAgent-RAG", version="1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(service: Service) -> HealthResponse:
        return service.health()

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
        try:
            return service.diagnose(request)
        except IndexNotReady as exc:
            raise HTTPException(503, str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(503, "The configured model provider is unavailable.") from exc

    return app
