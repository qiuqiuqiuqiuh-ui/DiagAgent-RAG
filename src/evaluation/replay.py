import json
from collections.abc import Sequence, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from src.agents.citations import CitationError, validate_citations
from src.agents.schemas import DiagnosisStatus, DiagnosticResult, NonBlank
from src.core.exceptions import ConfigurationError, CorpusError, ModelClientError
from src.core.llm import BaseLLMClient, create_llm_client
from src.core.settings import LLMSettings
from src.evaluation.dataset import EvalCase, EvalDataset
from src.evaluation.frozen import Mode
from src.evaluation.runner import judge_faithfulness, judge_gateway, judge_prompt_sha256
from src.ingestion.schemas import Chunk

Arm = Literal["without_request", "with_request"]
Report = TypeVar("Report", bound=BaseModel)


class _EvidenceRow(BaseModel):
    cited_evidence: tuple[Chunk, ...] | None = None


class _NestedRow(_EvidenceRow):
    case_id: str
    expected_status: DiagnosisStatus
    result: DiagnosticResult


class _PhaseBRow(DiagnosticResult):
    case_id: str
    expected_status: DiagnosisStatus
    cited_evidence: tuple[Chunk, ...]


class _EvidenceReport(BaseModel):
    index_version: NonBlank
    rows: tuple[_EvidenceRow, ...]


class _AnswerSummary(BaseModel):
    mode: Mode | None = None


class _AnswerReport(BaseModel):
    run_id: NonBlank
    model: NonBlank
    policy_sha256: str | None = None
    dataset_sha256: NonBlank
    index_version: NonBlank
    mode: Mode | None = None
    summary: _AnswerSummary | None = None
    rows: tuple[_NestedRow | _PhaseBRow, ...]


@dataclass(frozen=True)
class _FrozenAnswer:
    case: EvalCase
    result: DiagnosticResult
    evidence: tuple[Chunk, ...]


def _read_report(path: Path, schema: type[Report]) -> tuple[Report, str]:
    raw = path.read_bytes()
    try:
        report = schema.model_validate_json(raw)
    except ValidationError:
        raise CorpusError(f"Invalid frozen report structure in {path.name}") from None
    return report, sha256(raw).hexdigest()


def _load_answers(
    report: _AnswerReport,
    dataset: EvalDataset,
    evidence_reports: Sequence[_EvidenceReport],
    selected: Set[str] | None,
) -> list[_FrozenAnswer]:
    if report.dataset_sha256 != dataset.sha256:
        raise CorpusError("Answer report and dataset SHA-256 differ")
    if [row.case_id for row in report.rows] != [case.id for case in dataset.cases]:
        raise CorpusError("Frozen report case IDs or order differ from the dataset")
    for case, row in zip(dataset.cases, report.rows, strict=True):
        if row.expected_status != case.expected_status:
            raise CorpusError(f"{case.id}: frozen report and dataset expected statuses differ")

    chunks: dict[str, Chunk] = {}
    for source in (report, *evidence_reports):
        if source.index_version != report.index_version:
            raise CorpusError("Evidence and answer reports must use the same index version")
        for row in source.rows:
            for chunk in row.cited_evidence or ():
                if chunks.setdefault(chunk.chunk_id, chunk) != chunk:
                    raise CorpusError("Conflicting frozen evidence for the same chunk ID")

    answers = []
    for case, row in zip(dataset.cases, report.rows, strict=True):
        result = row.result if isinstance(row, _NestedRow) else row
        if result.status != "answered" or (selected is not None and case.id not in selected):
            continue
        evidence = row.cited_evidence
        if evidence is None:
            # Older smoke reports have only citation metadata in the answer row.
            try:
                evidence = tuple(chunks[citation.chunk_id] for citation in result.citations)
            except KeyError:
                raise CorpusError(f"{case.id}: missing frozen chunk for a citation") from None
        try:
            citations = validate_citations(result.answer, evidence)
        except CitationError:
            raise CorpusError(
                f"{case.id}: invalid inline citations or missing frozen chunk"
            ) from None
        if isinstance(row, _PhaseBRow):
            result = result.model_copy(update={"citations": citations})
        elif citations != result.citations:
            raise CorpusError(f"{case.id}: inline citations and frozen metadata differ")
        if len(evidence) != len(citations) or {chunk.chunk_id for chunk in evidence} != {
            citation.chunk_id for citation in citations
        }:
            raise CorpusError(f"{case.id}: expected one frozen chunk for each citation")
        answers.append(_FrozenAnswer(case, result, evidence))

    if not answers or (selected is not None and selected != {item.case.id for item in answers}):
        raise CorpusError("Every selected case must have a frozen answered result")
    return answers


def _score_answers(client: BaseLLMClient, answers: Sequence[_FrozenAnswer], paired: bool) -> dict:
    arms: tuple[Arm, ...] = ("without_request", "with_request") if paired else ("with_request",)
    scores: dict[Arm, list[float]] = {arm: [] for arm in arms}
    rows = []
    deltas = []
    for item in answers:
        outcomes = {}
        for arm in arms:
            start = perf_counter()
            try:
                score = judge_faithfulness(
                    client,
                    item.case.request.query,
                    item.result,
                    item.evidence,
                    request=item.case.request if arm == "with_request" else None,
                )
            except ModelClientError as exc:
                outcome = {
                    "score": None,
                    "error": "Judge call failed",
                    "status_code": exc.status_code,
                }
            else:
                if score is None:
                    raise ValueError("Judge replay requires a frozen answered result")
                scores[arm].append(score)
                outcome = {"score": score, "error": None, "status_code": None}
            outcome["elapsed_seconds"] = perf_counter() - start
            outcomes[arm] = outcome
        row = {"case_id": item.case.id, "arms": outcomes}
        if paired:
            before = outcomes["without_request"]["score"]
            after = outcomes["with_request"]["score"]
            delta = after - before if before is not None and after is not None else None
            row["paired_delta"] = delta
            if delta is not None:
                deltas.append(delta)
        rows.append(row)
    summary = {
        arm: {
            "attempted": len(answers),
            "scored": len(values),
            "failed": len(answers) - len(values),
            "mean": mean(values) if values else None,
        }
        for arm, values in scores.items()
    }
    result = {"summary": summary, "rows": rows}
    if paired:
        result["paired"] = {"scored": len(deltas), "mean_delta": mean(deltas) if deltas else None}
    return result


def replay_judge(
    judge_settings: LLMSettings,
    *,
    answers_path: Path,
    dataset: EvalDataset,
    output_path: Path,
    evidence_paths: Sequence[Path] = (),
    case_ids: Set[str] | None = None,
    paired: bool = False,
    allow_same_model: bool = False,
) -> dict:
    """Score frozen answers without opening an index or running the diagnostic workflow."""
    if judge_settings.provider == "mock":
        raise ConfigurationError("Judge replay requires a live LLM profile")
    gateway = judge_gateway(judge_settings.base_url)
    answers_report, answer_digest = _read_report(answers_path, _AnswerReport)
    if not allow_same_model and judge_settings.model.casefold() == answers_report.model.casefold():
        raise ConfigurationError("The judge model must differ from the frozen answer model")
    sources = [_read_report(path, _EvidenceReport) for path in evidence_paths]
    answers = _load_answers(answers_report, dataset, [report for report, _ in sources], case_ids)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite replay output {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    with create_llm_client(judge_settings) as client:
        report = {
            "run_id": uuid4().hex,
            "created_at": created_at.isoformat(),
            "judge_provider": client.provider,
            "judge_model": client.model,
            "judge_gateway": gateway,
            "temperature": judge_settings.temperature,
            "json_mode": judge_settings.json_mode,
            "judge_prompt_sha256": judge_prompt_sha256(),
            "answer_run_id": answers_report.run_id,
            "answer_model": answers_report.model,
            "answer_mode": answers_report.mode
            or (answers_report.summary.mode if answers_report.summary is not None else None),
            "answer_report_sha256": answer_digest,
            "answer_policy_sha256": answers_report.policy_sha256,
            "evidence_report_sha256": [digest for _, digest in sources],
            "dataset_sha256": dataset.sha256,
            "index_version": answers_report.index_version,
            **_score_answers(client, answers, paired),
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with output_path.open("x", encoding="utf-8") as output:
        output.write(payload)
    return report
