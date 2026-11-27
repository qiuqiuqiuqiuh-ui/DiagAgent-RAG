from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel

from src.agents.schemas import DiagnosisStatus, DiagnosticResult
from src.ingestion.schemas import Chunk

Mode = Literal["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rerank"]


class FrozenAnswerRow(BaseModel):
    case_id: str
    expected_status: DiagnosisStatus
    result: DiagnosticResult
    cited_evidence: tuple[Chunk, ...]


class FrozenModeReport(BaseModel):
    run_id: str
    created_at: AwareDatetime
    model: str
    dataset_sha256: str
    index_version: str
    mode: Mode
    rows: tuple[FrozenAnswerRow, ...]


def write_frozen_mode(
    directory: Path,
    *,
    created_at: datetime,
    model: str,
    dataset_sha256: str,
    index_version: str,
    mode: Mode,
    rows: list[FrozenAnswerRow],
) -> Path:
    run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    report = FrozenModeReport(
        run_id=run_id,
        created_at=created_at,
        model=model,
        dataset_sha256=dataset_sha256,
        index_version=index_version,
        mode=mode,
        rows=tuple(rows),
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}_{mode}_answers.json"
    with path.open("x", encoding="utf-8") as output:
        output.write(report.model_dump_json(indent=2) + "\n")
    return path
