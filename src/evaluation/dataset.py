from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.agents.schemas import DiagnosisStatus, DiagnosticRequest, Intent, NonBlank
from src.core.exceptions import CorpusError

Category = Literal["single_hop", "multi_hop", "sensor_threshold", "unanswerable", "out_of_scope"]


class ExpectedTool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["fault_code", "sensor_threshold"]
    target: NonBlank
    status: Literal["found", "normal", "warning", "critical", "unknown", "ambiguous"]
    equipment_id: NonBlank | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "ExpectedTool":
        if self.status in ("unknown", "ambiguous"):
            return self
        if (self.kind == "fault_code") != (self.status == "found"):
            raise ValueError("Expected tool status does not match the tool kind")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    category: Category
    request: DiagnosticRequest
    expected_intent: Intent
    expected_status: DiagnosisStatus
    relevant_chunks: tuple[NonBlank, ...] = ()
    expected_tools: tuple[ExpectedTool, ...] = ()

    @model_validator(mode="after")
    def validate_labels(self) -> "EvalCase":
        expected_status = {
            "unanswerable": "insufficient_evidence",
            "out_of_scope": "out_of_scope",
        }.get(self.category, "answered")
        if self.expected_status != expected_status:
            raise ValueError("Expected status does not match the case category")
        if (self.expected_intent == "out_of_scope") != (self.expected_status == "out_of_scope"):
            raise ValueError("Out-of-scope status and intent must agree")
        if (self.expected_status == "answered") != bool(self.relevant_chunks):
            raise ValueError("Every answerable case requires annotated evidence")
        if len(set(self.relevant_chunks)) != len(self.relevant_chunks):
            raise ValueError("Relevant chunk IDs must be unique")
        return self


@dataclass(frozen=True)
class EvalDataset:
    name: str
    sha256: str
    cases: tuple[EvalCase, ...]


def load_dataset(path: Path) -> EvalDataset:
    try:
        content = path.read_bytes().replace(b"\r\n", b"\n")
    except OSError as exc:
        raise CorpusError(f"Could not read evaluation dataset {path}") from exc
    cases = []
    ids = set()
    for number, line in enumerate(content.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line.removeprefix(b"\xef\xbb\xbf"))
        except ValidationError as exc:
            raise CorpusError(f"{path.name}:{number}: invalid evaluation case") from exc
        if case.id in ids:
            raise CorpusError(f"{path.name}:{number}: duplicate case ID {case.id}")
        ids.add(case.id)
        cases.append(case)
    if not cases:
        raise CorpusError(f"Evaluation dataset {path} does not contain any cases")
    return EvalDataset(path.name, sha256(content).hexdigest(), tuple(cases))
