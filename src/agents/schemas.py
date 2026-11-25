from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

Intent = Literal["fault_diagnosis", "parameter_lookup", "out_of_scope"]
DiagnosisStatus = Literal["answered", "insufficient_evidence", "out_of_scope"]
NodeName = Literal["intent_router", "retriever", "diagnostic_tool", "synthesizer", "critic"]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Reading = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class DiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: NonBlank
    equipment_id: NonBlank | None = None
    fault_code: NonBlank | None = None
    sensor_readings: dict[NonBlank, Reading] = Field(default_factory=dict)

    @field_validator("sensor_readings", mode="before")
    @classmethod
    def reject_duplicate_labels(cls, value: object) -> object:
        if isinstance(value, dict):
            labels = [key.strip().casefold() for key in value if isinstance(key, str)]
            if len(labels) != len(set(labels)):
                raise ValueError("Sensor labels must be unique after normalization")
        return value


class FaultCodeRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment_id: NonBlank
    fault_code: NonBlank
    severity: NonBlank
    meaning: NonBlank
    first_check: NonBlank
    escalation: NonBlank


class SensorThresholdRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment_id: NonBlank
    sensor: NonBlank
    unit: NonBlank
    warning_threshold: FiniteFloat
    critical_threshold: FiniteFloat
    direction: Literal["high", "low"]

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "SensorThresholdRule":
        ordered = (
            self.warning_threshold < self.critical_threshold
            if self.direction == "high"
            else self.warning_threshold > self.critical_threshold
        )
        if not ordered:
            raise ValueError("Critical threshold must be beyond the warning threshold")
        return self


class FaultCodeFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["fault_code"] = "fault_code"
    fault_code: str
    status: Literal["found", "unknown", "ambiguous"]
    rule: FaultCodeRule | None = None


class SensorFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["sensor_threshold"] = "sensor_threshold"
    sensor: str
    reading: Reading
    status: Literal["normal", "warning", "critical", "unknown", "ambiguous"]
    rule: SensorThresholdRule | None = None


ToolFinding = Annotated[FaultCodeFinding | SensorFinding, Field(discriminator="kind")]


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    source_path: str
    section_title: str
    page_number: int | None = None


class CriticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    missing_aspects: tuple[NonBlank, ...] = ()
    refined_query: NonBlank | None = None


class NodeTrace(BaseModel):
    node: NodeName
    latency_ms: FiniteFloat = Field(ge=0)
    retry_count: int = Field(ge=0)
    top_k: int | None = Field(default=None, gt=0)


class StateTransition(BaseModel):
    from_node: NodeName | Literal["START"]
    to_node: NodeName | Literal["END"]


class DiagnosticTrace(BaseModel):
    nodes: tuple[NodeTrace, ...]
    transitions: tuple[StateTransition, ...]
    retry_count: int = Field(ge=0)


class DiagnosticResult(BaseModel):
    status: DiagnosisStatus
    intent: Intent | None
    answer: NonBlank
    citations: tuple[Citation, ...] = ()
    tool_findings: tuple[ToolFinding, ...] = ()
    missing_aspects: tuple[NonBlank, ...] = ()
    escalation_recommendation: NonBlank | None = None
    trace: DiagnosticTrace | None = None
