from typing import Literal

from pydantic import BaseModel, Field

from src.agents.schemas import DiagnosticRequest, DiagnosticResult, SensorThresholdRule


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


class DomainMetadata(BaseModel):
    name: str
    language: str


class MetaResponse(BaseModel):
    domain: DomainMetadata
    mode: Literal["mock", "live"]
    llm_model: str
    embedding_model: str
    reranker_model: str
    equipment_ids: tuple[str, ...] = ()
    fault_codes: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    sensors: tuple[SensorThresholdRule, ...] = ()


class ChunkDetailResponse(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    section_title: str
    page_number: int | None
    text: str
