from typing import Literal

from pydantic import BaseModel

from src.agents.schemas import DiagnosticRequest, DiagnosticResult


class DiagnoseRequest(DiagnosticRequest):
    bypass_cache: bool = False
    include_trace: bool = True

    def diagnostic_request(self) -> DiagnosticRequest:
        return DiagnosticRequest.model_validate(
            self.model_dump(exclude={"bypass_cache", "include_trace"})
        )


class DiagnoseResponse(DiagnosticResult):
    cache_hit: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    index_version: str | None = None
    chunk_count: int = 0
