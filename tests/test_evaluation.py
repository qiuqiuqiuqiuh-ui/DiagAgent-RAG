import ast
import json
import runpy
from collections.abc import Sequence
from dataclasses import replace
from math import log2
from pathlib import Path

import pytest

from src.agents.citations import validate_citations
from src.agents.schemas import DiagnosticRequest, DiagnosticResult, FaultCodeFinding
from src.agents.tools import DiagnosticTools
from src.core.embedding import create_embedding_client
from src.core.exceptions import ConfigurationError, CorpusError, ModelClientError
from src.core.llm import ChatMessage, MockLLMClient
from src.core.settings import LLMSettings, load_settings
from src.evaluation.dataset import EvalCase, load_dataset
from src.evaluation.frozen import FrozenModeReport
from src.evaluation.metrics import score_case, score_ranking, summarize
from src.evaluation.report import render_markdown, write_reports
from src.evaluation.runner import (
    EvaluationReport,
    _RecordingRetriever,
    evaluate,
    judge_faithfulness,
    judge_prompt_sha256,
)
from src.ingestion.corpus import read_corpus
from src.ingestion.pipeline import build_index
from src.ingestion.schemas import Chunk
from src.retrieval.schemas import SearchHit

ROOT = Path(__file__).resolve().parents[1]


def test_judge_prompt_matches_phase_b() -> None:
    assert (
        judge_prompt_sha256() == "9b72030702cfaede7c096ebc8ba9a0c959ef138fe637105024eeedbcbddb11ef"
    )


def test_golden_dataset_sizes_labels_and_tool_expectations() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    tools = DiagnosticTools.from_corpus(settings.paths.corpus_dir)
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    smoke = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    bench = load_dataset(ROOT / "data" / "eval" / "golden_bench.jsonl")
    assert len(smoke.cases) == 15
    assert sum(case.expected_status != "answered" for case in smoke.cases) == 3
    assert len(bench.cases) >= 50
    assert (
        0.13
        <= sum(case.expected_status != "answered" for case in bench.cases) / len(bench.cases)
        <= 0.17
    )
    assert {case.category for case in bench.cases} == {
        "single_hop",
        "multi_hop",
        "sensor_threshold",
        "unanswerable",
        "out_of_scope",
    }
    for case in (*smoke.cases, *bench.cases):
        assert set(case.relevant_chunks) <= chunk_ids
        verdicts = score_case(
            case,
            DiagnosticResult(
                status=case.expected_status,
                intent=case.expected_intent,
                answer="fixture",
                tool_findings=tools.evaluate(case.request),
            ),
            (),
            (),
            k=5,
            latency_ms=1.0,
        )
        assert verdicts.tools_correct in (None, True), case.id


def test_dataset_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    original = (
        (ROOT / "data" / "eval" / "golden_smoke.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{original}\n{original}\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="duplicate case ID"):
        load_dataset(path)


def test_ranking_metrics_with_duplicate_ranks_and_missing_evidence() -> None:
    score = score_ranking(["x", "a", "a", "b"], {"a", "b", "c"}, 4)
    assert score is not None
    assert score.recall_at_k == pytest.approx(2 / 3)
    assert score.mrr_at_k == 0.5
    assert score.ndcg_at_k == pytest.approx((1 / log2(3) + 1 / log2(5)) / (1 + 1 / log2(3) + 0.5))
    assert score_ranking(["x"], set(), 5) is None
    assert score_ranking(["x"], {"a"}, 1).ndcg_at_k == 0


def test_case_citations_metric_aggregation_and_optional_judge() -> None:
    case = EvalCase(
        id="test-case",
        category="single_hop",
        request={"query": "What caused the alarm?"},
        expected_intent="fault_diagnosis",
        expected_status="answered",
        relevant_chunks=("f.md#s0#c0",),
    )
    chunk = Chunk(
        doc_id="f.md",
        source_path="f.md",
        section_title="Alarm",
        text="An overcurrent alarm",
        section_index=0,
        chunk_index=0,
        chunk_id="f.md#s0#c0",
    )
    answer = "An overcurrent alarm [f.md#s0#c0]."
    result = DiagnosticResult(
        status="answered",
        intent="fault_diagnosis",
        answer=answer,
        citations=validate_citations(answer, [chunk]),
    )

    def respond(messages: Sequence[ChatMessage]) -> str:
        assert "request" not in json.loads(messages[1].content)
        return '```json\n{"faithfulness":0.75}\n```'

    judge = MockLLMClient(responder=respond)
    faithfulness = judge_faithfulness(judge, case.request.query, result, [chunk])
    assert faithfulness == 0.75
    supported = score_case(
        case, result, [chunk], [chunk], k=5, latency_ms=2, faithfulness=faithfulness
    )
    mismatched = score_case(case, result, [chunk], [], k=5, latency_ms=8)
    report = summarize([supported, mismatched])
    assert report.recall_at_k == 1.0
    assert report.citation_coverage == report.citation_validity == 0.5
    assert report.latency_p50_ms == 5 and report.latency_p95_ms == pytest.approx(7.7)
    assert report.faithfulness == 0.75 and report.judge_samples == 1


@pytest.mark.parametrize("readings", [{}, {"belt_speed": 0}])
def test_judge_receives_diagnostic_request_context(readings: dict[str, int]) -> None:
    request = DiagnosticRequest(
        query="What does ERR-402 mean?",
        equipment_id="CV-40",
        fault_code="ERR-402",
        sensor_readings=readings,
    )
    result = DiagnosticResult(status="answered", intent="fault_diagnosis", answer="Alarm details.")

    def respond(messages: Sequence[ChatMessage]) -> str:
        assert "what the user provided or omitted" in messages[0].content
        payload = json.loads(messages[1].content)
        assert payload["query"] == request.query
        assert payload["request"] == {
            "equipment_id": "CV-40",
            "fault_code": "ERR-402",
            "sensor_readings": readings,
        }
        return '{"faithfulness":1.0}'

    judge = MockLLMClient(responder=respond)
    assert judge_faithfulness(judge, request.query, result, [], request=request) == 1.0


def test_scoring_rejects_extra_and_reused_findings() -> None:
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    case = dataset.cases[0]
    tools = DiagnosticTools.from_corpus(ROOT / "data" / "raw")
    findings = tools.evaluate(case.request)
    extra = FaultCodeFinding(fault_code="BOGUS-999", status="unknown")
    result = DiagnosticResult(
        status="answered", intent="fault_diagnosis", answer="fixture", tool_findings=findings
    )
    assert score_case(case, result, (), (), k=5, latency_ms=1).tools_correct is True

    extra_result = result.model_copy(update={"tool_findings": (*findings, extra)})
    assert score_case(case, extra_result, (), (), k=5, latency_ms=1).tools_correct is False
    duplicate_labels = case.model_copy(update={"expected_tools": case.expected_tools * 2})
    assert (
        score_case(duplicate_labels, extra_result, (), (), k=5, latency_ms=1).tools_correct is False
    )


def test_retrieval_attempt_denominator_counts_empty_searches_and_preserves_legacy_unknown() -> None:
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    answerable = dataset.cases[0]
    unanswerable = next(case for case in dataset.cases if case.category == "unanswerable")
    result = DiagnosticResult(status="insufficient_evidence", intent=None, answer="fixture")
    routed = result.model_copy(update={"intent": "fault_diagnosis"})
    skipped = score_case(answerable, result, None, (), k=5, latency_ms=1)
    attempted = score_case(answerable, routed, (), (), k=5, latency_ms=1)
    without_gold = score_case(unanswerable, routed, (), (), k=5, latency_ms=1)

    assert skipped.retrieval_attempted is False
    assert attempted.retrieval_attempted is True and attempted.retrieved_ids == ()
    assert without_gold.retrieval_attempted is True and without_gold.ranking is None
    metrics = summarize([skipped, attempted, without_gold])
    assert metrics.retrieval_samples == 2 and metrics.retrieval_attempted_samples == 1
    assert metrics.recall_at_k == 0.0

    legacy = attempted.model_dump(exclude={"retrieval_attempted"})
    restored = type(attempted).model_validate(legacy)
    assert restored.retrieval_attempted is None
    assert summarize([skipped, restored]).retrieval_attempted_samples is None


def test_recording_retriever_clears_evidence_on_failed_attempt() -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    chunk = read_corpus(settings.paths.corpus_dir, settings.ingestion).chunks[0]

    def search(query: str, k: int) -> list[SearchHit]:
        if query == "fail":
            raise ModelClientError("retrieval failed")
        return [SearchHit(chunk=chunk, score=1.0)]

    capture = _RecordingRetriever(search)
    capture.search("ok")
    assert capture.first == capture.last == (chunk,)
    with pytest.raises(ModelClientError):
        capture.search("fail")
    assert capture.first == (chunk,)
    assert capture.last == ()

    capture.reset()
    with pytest.raises(ModelClientError):
        capture.search("fail")
    capture.search("ok")
    assert capture.first == ()
    assert capture.last == (chunk,)


def test_evaluation_preserves_full_report_when_one_judge_call_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    settings.paths.index_dir = tmp_path / "index"
    with create_embedding_client(settings.embedding) as embedder:
        build_index(settings, embedder)
    settings.llm.provider = "openai_compatible"
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    dataset = replace(dataset, cases=dataset.cases[:2])
    requests = {case.request.query: case.request for case in dataset.cases}
    judge_calls = 0

    def respond(messages: Sequence[ChatMessage]) -> str:
        nonlocal judge_calls
        prompt = messages[0].content
        if '"faithfulness"' in prompt:
            payload = json.loads(messages[1].content)
            assert payload["request"] == requests[payload["query"]].model_dump(
                mode="json", exclude={"query"}
            )
            judge_calls += 1
            if judge_calls == 1:
                raise ModelClientError("provider limited the judge", status_code=429)
            return '```json\n{"faithfulness":0.9}\n```'
        if "Classify the request" in prompt:
            return '{"intent":"fault_diagnosis"}'
        if "Draft a concise" in prompt:
            chunk_id = json.loads(messages[-1].content)["chunks"][0]["chunk_id"]
            return json.dumps({"answer": f"Follow the cited procedure [{chunk_id}]."})
        return '{"sufficient":true,"missing_aspects":[],"refined_query":null}'

    class ScriptedLiveClient(MockLLMClient):
        provider = "openai_compatible"

        def close(self) -> None:
            closed.append(self)

    clients = []
    closed = []

    def create_client(config: LLMSettings) -> ScriptedLiveClient:
        assert config is settings.llm
        client = ScriptedLiveClient(model=config.model, responder=respond)
        clients.append(client)
        return client

    monkeypatch.setattr("src.evaluation.runner.create_llm_client", create_client)
    progress: list[tuple[str, int, float]] = []
    report = evaluate(
        settings,
        dataset,
        judge=True,
        on_mode_complete=lambda mode, elapsed: progress.append(
            (mode.mode, mode.metrics.samples, elapsed)
        ),
    )
    json_path, markdown_path = write_reports(report, tmp_path / "reports")
    reloaded = EvaluationReport.model_validate_json(json_path.read_bytes())
    assert markdown_path.is_file()
    assert len(reloaded.modes) == 4 and all(len(mode.cases) == 2 for mode in reloaded.modes)
    assert len(clients) == 2 and clients[0] is not clients[1] and closed == clients[::-1]
    assert reloaded.providers["judge"] == f"openai_compatible:{settings.llm.model}"
    assert judge_calls == 8
    assert [mode.metrics.judge_samples for mode in reloaded.modes] == [1, 2, 2, 2]
    assert reloaded.modes[0].cases[0].faithfulness is None
    assert all(mode.metrics.faithfulness == pytest.approx(0.9) for mode in reloaded.modes)
    assert all(mode.metrics.retrieval_attempted_samples == 2 for mode in reloaded.modes)
    assert "| Retrieval attempts (gold cases) | 2 | 2 | 2 | 2 |" in render_markdown(reloaded)
    assert "| Judge samples | 1 | 2 | 2 | 2 |" in render_markdown(reloaded)
    legacy = reloaded.model_dump()
    for mode in legacy["modes"]:
        del mode["metrics"]["retrieval_attempted_samples"]
        for case in mode["cases"]:
            del case["retrieval_attempted"]
    restored = EvaluationReport.model_validate(legacy)
    assert "Retrieval attempts (gold cases)" not in render_markdown(restored)
    assert [mode for mode, _, _ in progress] == [mode.mode for mode in reloaded.modes]
    assert all(count == 2 and elapsed >= 0.0 for _, count, elapsed in progress)
    failures = [record for record in caplog.records if record.message == "evaluation.judge_failed"]
    assert len(failures) == 1 and failures[0].event_fields["status_code"] == 429


def test_evaluation_uses_only_the_independent_judge_for_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    settings.paths.index_dir = tmp_path / "index"
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    with pytest.raises(ConfigurationError, match="requires judging"):
        evaluate(settings, dataset, require_heterogeneous_judge=True)
    with pytest.raises(ConfigurationError, match="live LLM"):
        evaluate(settings, dataset, judge=True)
    settings.judge_llm = LLMSettings(
        provider="openai_compatible", model=settings.llm.model.upper(), api_key="judge-test-key"
    )
    with pytest.raises(ConfigurationError, match="must differ"):
        evaluate(settings, dataset, judge=True, require_heterogeneous_judge=True)
    settings.judge_llm.model = "independent-judge"
    with create_embedding_client(settings.embedding) as embedder:
        build_index(settings, embedder)
    closed = []
    calls = []
    configs = []

    def respond(messages: Sequence[ChatMessage]) -> str:
        assert messages[0].content.startswith('Return a JSON object {"faithfulness"')
        calls.append(messages)
        return '{"faithfulness":0.8}'

    class AnswerClient(MockLLMClient):
        def close(self) -> None:
            closed.append(self.model)

    class JudgeClient(AnswerClient):
        provider = "openai_compatible"

    def create_client(config: LLMSettings) -> MockLLMClient:
        configs.append(config)
        if config is settings.judge_llm:
            return JudgeClient(config.model, responder=respond)
        return AnswerClient(config.model)

    monkeypatch.setattr("src.evaluation.runner.create_llm_client", create_client)
    report = evaluate(settings, dataset, judge=True, require_heterogeneous_judge=True)
    assert configs == [settings.llm, settings.judge_llm]
    assert closed == [settings.judge_llm.model, settings.llm.model]
    assert report.run_mode == "mock"
    assert report.providers["judge"] == "openai_compatible:independent-judge"
    assert "judge-test-key" not in report.model_dump_json() + render_markdown(report)
    assert len(calls) == sum(mode.metrics.judge_samples for mode in report.modes) > 0


def test_evaluate_cli_saves_replayable_answers_for_every_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_dir = tmp_path / "index"
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("DIAGAGENT_PATHS__INDEX_DIR", str(index_dir))
    monkeypatch.setenv("DIAGAGENT_PATHS__REPORTS_DIR", str(reports_dir))
    monkeypatch.setenv("DIAGAGENT_PATHS__CORPUS_DIR", str(ROOT / "data" / "raw"))
    settings = load_settings(project_root=ROOT, env_file=None)
    with create_embedding_client(settings.embedding) as embedder:
        build_index(settings, embedder)
    dataset_path = ROOT / "data" / "eval" / "golden_smoke.jsonl"
    dataset = load_dataset(dataset_path)
    evaluate_main = runpy.run_path(str(ROOT / "scripts" / "evaluate.py"))["main"]
    with pytest.raises(SystemExit) as failure:
        evaluate_main(["--require-heterogeneous-judge"])
    assert failure.value.code == 2
    args = [
        "--project-root",
        str(tmp_path),
        "--config",
        str(ROOT / "configs" / "default.yaml"),
        "--dataset",
        str(dataset_path),
        "--save-answers",
    ]
    assert evaluate_main(args) == 0
    frozen = [
        FrozenModeReport.model_validate_json(path.read_bytes())
        for path in reports_dir.glob("*_answers.json")
    ]
    assert {report.mode for report in frozen} == {
        "dense_only",
        "bm25_only",
        "hybrid_rrf",
        "hybrid_rerank",
    }
    assert len({report.run_id for report in frozen}) == 1
    assert len({report.index_version for report in frozen}) == 1
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    for report in frozen:
        assert report.model == settings.llm.model and report.dataset_sha256 == dataset.sha256
        assert [(row.case_id, row.expected_status) for row in report.rows] == [
            (case.id, case.expected_status) for case in dataset.cases
        ]
        assert any(row.result.tool_findings for row in report.rows)
        for row in report.rows:
            assert row.result.trace is not None and row.result.trace.nodes
            assert all(chunks[chunk.chunk_id] == chunk for chunk in row.cited_evidence)
            if row.result.status == "answered":
                assert (
                    validate_citations(row.result.answer, row.cited_evidence)
                    == row.result.citations
                )
            else:
                assert not row.cited_evidence


def test_colab_notebook_matches_script_and_has_clean_cells() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "colab_gpu_pipeline.ipynb").read_text(encoding="utf-8")
    )
    script = (ROOT / "notebooks" / "colab_gpu_pipeline.py").read_text(encoding="utf-8")
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU"
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 5 and all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    cells_code = "\n\n".join("".join(cell["source"]) for cell in cells)
    assert ast.dump(ast.parse(cells_code)) == ast.dump(ast.parse(script))
