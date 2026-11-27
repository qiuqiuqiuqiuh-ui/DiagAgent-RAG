from math import isclose, isfinite
from statistics import fmean

from pydantic import BaseModel, Field

from src.core.exceptions import ConfigurationError
from src.evaluation.frozen import Mode
from src.evaluation.runner import EvaluationReport, ModeReport


class ReplayArm(BaseModel):
    score: float | None = Field(ge=0, le=1, allow_inf_nan=False, strict=True)


class ReplayRow(BaseModel):
    case_id: str
    arms: dict[str, ReplayArm]


class GateReplayReport(BaseModel):
    answer_run_id: str
    answer_model: str
    answer_mode: Mode
    dataset_sha256: str
    index_version: str
    rows: tuple[ReplayRow, ...]


def _faithfulness(
    report: EvaluationReport, mode: ModeReport, replay: GateReplayReport | None
) -> tuple[float | None, int, int]:
    answered = {case.id for case in mode.cases if case.status == "answered"}
    if replay is None:
        scores = [case.faithfulness for case in mode.cases if case.id in answered]
    else:
        expected_run_id = report.created_at.strftime("%Y%m%dT%H%M%S%fZ")
        if (
            replay.dataset_sha256 != report.dataset_sha256
            or replay.index_version != report.index_version
            or replay.answer_run_id != expected_run_id
            or replay.answer_mode != mode.mode
            or replay.answer_model != report.providers.get("llm", "").partition(":")[2]
        ):
            raise ConfigurationError(
                "Replay dataset, index, answer run, mode and model must match the evaluation"
            )
        case_ids = [row.case_id for row in replay.rows]
        if len(case_ids) != len(set(case_ids)) or not set(case_ids) <= answered:
            raise ConfigurationError(
                "Replay rows must reference unique answered cases in the evaluation"
            )
        if any("with_request" not in row.arms for row in replay.rows):
            raise ConfigurationError("Replay rows must include the with_request score")
        scores = [row.arms["with_request"].score for row in replay.rows]

    valid = [score for score in scores if score is not None and isfinite(score) and 0 <= score <= 1]
    return (fmean(valid) if valid else None), len(valid), len(answered)


def check_eval_gate(
    report: EvaluationReport, mode_name: Mode, *, replay: GateReplayReport | None = None
) -> dict:
    selected = [mode for mode in report.modes if mode.mode == mode_name]
    if len(selected) != 1:
        raise ConfigurationError(
            "Evaluation report must contain the selected retrieval mode exactly once"
        )
    (mode,) = selected
    metrics = mode.metrics
    if (
        metrics.samples != len(mode.cases)
        or len({case.id for case in mode.cases}) != metrics.samples
    ):
        raise ConfigurationError("Evaluation sample count or case IDs are inconsistent")
    faithfulness, judge_samples, answered_samples = _faithfulness(report, mode, replay)
    thresholds = (
        ("status_accuracy", metrics.status_accuracy, 0.98, metrics.samples),
        ("abstention_accuracy", metrics.abstention_accuracy, 1.0, metrics.samples),
        ("citation_validity", metrics.citation_validity, 1.0, metrics.citation_samples),
        ("tool_accuracy", metrics.tool_accuracy, 1.0, metrics.tool_samples),
        ("faithfulness", faithfulness, 0.95, judge_samples),
        ("recall_at_k", metrics.recall_at_k, 0.95, metrics.retrieval_samples),
    )
    results = {}
    for name, value, threshold, samples in thresholds:
        valid = value is not None and isfinite(value) and 0 <= value <= 1
        meets_threshold = valid and (
            value >= threshold
            or (threshold < 1 and isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12))
        )
        results[name] = {
            "value": value if valid else None,
            "minimum": threshold,
            "samples": samples,
            "passed": meets_threshold and samples > 0,
        }
    faith = results["faithfulness"]
    faith["required_samples"] = answered_samples
    faith["source"] = "replay" if replay is not None else "evaluation"
    faith["passed"] = faith["passed"] and judge_samples == answered_samples
    return {
        "status": "PASS" if all(check["passed"] for check in results.values()) else "FAIL",
        "mode": mode_name,
        "dataset_sha256": report.dataset_sha256,
        "index_version": report.index_version,
        "checks": results,
    }
