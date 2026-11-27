import json
from datetime import datetime, timezone
from pathlib import Path
from runpy import run_path

import pytest

from src.core.exceptions import ConfigurationError
from src.evaluation.gate import GateReplayReport, check_eval_gate
from src.evaluation.metrics import CaseResult, RankingMetrics, summarize
from src.evaluation.runner import EvaluationReport, ModeReport

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report() -> EvaluationReport:
    answer = CaseResult(
        id="answer-01",
        category="single_hop",
        expected_status="answered",
        status="answered",
        intent="fault_diagnosis",
        retrieved_ids=("manual#s0#c0",),
        cited_ids=("manual#s0#c0",),
        ranking=RankingMetrics(recall_at_k=0.95, mrr_at_k=1.0, ndcg_at_k=1.0),
        citation_coverage=1.0,
        citation_valid=True,
        routing_correct=True,
        tools_correct=True,
        abstention_correct=True,
        latency_ms=1.0,
        faithfulness=0.95,
    )
    refusal = CaseResult(
        id="refuse-01",
        category="unanswerable",
        expected_status="insufficient_evidence",
        status="insufficient_evidence",
        intent="parameter_lookup",
        retrieved_ids=(),
        cited_ids=(),
        ranking=None,
        citation_coverage=None,
        citation_valid=None,
        routing_correct=True,
        tools_correct=None,
        abstention_correct=True,
        latency_ms=1.0,
    )
    cases = (answer, refusal)
    return EvaluationReport(
        created_at=datetime(2026, 11, 27, tzinfo=timezone.utc),
        run_mode="live",
        dataset="golden.jsonl",
        dataset_sha256="dataset",
        index_version="index",
        top_k=5,
        providers={"llm": "openai_compatible:answer:version"},
        modes=(ModeReport(mode="hybrid_rerank", metrics=summarize(cases), cases=cases),),
    )


@pytest.fixture
def replay_data() -> dict:
    return {
        "answer_run_id": "20261127T000000000000Z",
        "answer_mode": "hybrid_rerank",
        "answer_model": "answer:version",
        "dataset_sha256": "dataset",
        "index_version": "index",
        "rows": [
            {
                "case_id": "answer-01",
                "arms": {"with_request": {"score": 0.97}, "without_request": {"score": 0.1}},
            }
        ],
    }


def test_evaluation_and_replay_clear_thresholds_with_correct_denominators(
    report: EvaluationReport, replay_data: dict
) -> None:
    direct = check_eval_gate(report, "hybrid_rerank")
    assert direct["status"] == "PASS"
    assert direct["checks"]["faithfulness"]["value"] == 0.95
    replay = GateReplayReport.model_validate(replay_data)
    checked = check_eval_gate(report, "hybrid_rerank", replay=replay)
    assert checked["status"] == "PASS"
    assert checked["checks"]["faithfulness"] == {
        "value": 0.97,
        "minimum": 0.95,
        "samples": 1,
        "passed": True,
        "required_samples": 1,
        "source": "replay",
    }


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("status_accuracy", 0.979),
        ("abstention_accuracy", 0.99),
        ("citation_validity", 0.99),
        ("tool_accuracy", None),
        ("recall_at_k", 0.949),
    ],
)
def test_gate_fails_below_threshold_or_unmeasured_metric(
    report: EvaluationReport, metric: str, value: float | None
) -> None:
    report.modes[0].metrics = report.modes[0].metrics.model_copy(update={metric: value})
    checked = check_eval_gate(report, "hybrid_rerank")
    assert checked["status"] == "FAIL"
    assert checked["checks"][metric]["passed"] is False


@pytest.mark.parametrize("score", [None, 0.94])
def test_missing_or_low_judge_score_cannot_pass(
    report: EvaluationReport, replay_data: dict, score: float | None
) -> None:
    report.modes[0].cases[0].faithfulness = score
    assert check_eval_gate(report, "hybrid_rerank")["status"] == "FAIL"
    replay_data["rows"][0]["arms"]["with_request"]["score"] = score
    replay = GateReplayReport.model_validate(replay_data)
    checked = check_eval_gate(report, "hybrid_rerank", replay=replay)
    assert checked["status"] == "FAIL"
    assert checked["checks"]["faithfulness"]["required_samples"] == 1


def test_partial_replay_uses_full_answered_denominator(
    report: EvaluationReport, replay_data: dict
) -> None:
    mode = report.modes[0]
    mode.cases += (mode.cases[0].model_copy(update={"id": "answer-02"}),)
    mode.metrics = summarize(mode.cases)
    replay = GateReplayReport.model_validate(replay_data)
    result = check_eval_gate(report, "hybrid_rerank", replay=replay)
    assert result["status"] == "FAIL"
    assert result["checks"]["faithfulness"]["samples"] == 1
    assert result["checks"]["faithfulness"]["required_samples"] == 2


def test_gate_thresholds_tolerate_rounding_but_require_perfect_citations(
    report: EvaluationReport,
) -> None:
    mode = report.modes[0]
    mode.cases[0].faithfulness = 0.9499999999999998
    assert check_eval_gate(report, "hybrid_rerank")["status"] == "PASS"
    mode.metrics.citation_validity = 0.9999999999999999
    assert (
        check_eval_gate(report, "hybrid_rerank")["checks"]["citation_validity"]["passed"] is False
    )


@pytest.mark.parametrize(
    "field", ["answer_run_id", "answer_model", "answer_mode", "dataset_sha256", "index_version"]
)
def test_gate_rejects_replay_from_another_run(
    report: EvaluationReport, replay_data: dict, field: str
) -> None:
    replay_data[field] = "bm25_only" if field == "answer_mode" else "mismatch"
    with pytest.raises(ConfigurationError, match="must match"):
        check_eval_gate(
            report, "hybrid_rerank", replay=GateReplayReport.model_validate(replay_data)
        )


def test_gate_cli_reports_pass_and_rejects_invalid_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], report: EvaluationReport
) -> None:
    cli = run_path(str(ROOT / "scripts" / "check_eval_gate.py"))["main"]
    path = tmp_path / "eval.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    assert cli(["--report", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    path.write_text("invalid JSON", encoding="utf-8")
    assert cli(["--report", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_gate_allow_no_tools_supports_manual_only_corpora_without_relaxing_measured_tools(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], report: EvaluationReport
) -> None:
    mode = report.modes[0]
    manual_only_cases = tuple(
        case.model_copy(update={"tools_correct": None}) for case in mode.cases
    )
    manual_only = report.model_copy(
        update={
            "modes": (
                ModeReport(
                    mode="hybrid_rerank",
                    metrics=summarize(manual_only_cases),
                    cases=manual_only_cases,
                ),
            )
        }
    )
    assert check_eval_gate(manual_only, "hybrid_rerank")["status"] == "FAIL"
    passed = check_eval_gate(manual_only, "hybrid_rerank", allow_no_tools=True)
    assert passed["status"] == "PASS"
    assert passed["checks"]["tool_accuracy"] == {
        "value": None,
        "minimum": 1.0,
        "samples": 0,
        "skipped": True,
        "passed": True,
    }

    cli = run_path(str(ROOT / "scripts" / "check_eval_gate.py"))["main"]
    path = tmp_path / "manual_only.json"
    path.write_text(manual_only.model_dump_json(), encoding="utf-8")
    assert cli(["--report", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert cli(["--report", str(path), "--allow-no-tools"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    failing_tools = report.model_copy(deep=True)
    failing_tools.modes[0].metrics = failing_tools.modes[0].metrics.model_copy(
        update={"tool_accuracy": 0.5}
    )
    assert check_eval_gate(failing_tools, "hybrid_rerank", allow_no_tools=True)["status"] == "FAIL"
