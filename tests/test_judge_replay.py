import json
import runpy
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from src.agents.citations import validate_citations
from src.agents.schemas import DiagnosticResult
from src.core.exceptions import ConfigurationError, CorpusError
from src.core.llm import ChatMessage, MockLLMClient, OpenAICompatibleLLMClient
from src.core.settings import LLMSettings, load_settings
from src.evaluation.dataset import EvalDataset, load_dataset
from src.evaluation.replay import replay_judge
from src.evaluation.runner import judge_prompt_sha256
from src.ingestion.corpus import read_corpus

ROOT = Path(__file__).resolve().parents[1]
SECRET = "test-replay-api-key"


@pytest.fixture
def fixture_data() -> tuple[EvalDataset, dict]:
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    dataset = replace(dataset, cases=dataset.cases[:2])
    settings = load_settings(project_root=ROOT, env_file=None)
    corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    rows = []
    for case in dataset.cases:
        evidence = chunks[case.relevant_chunks[0]]
        answer = f"Follow the manual [{evidence.chunk_id}]."
        result = DiagnosticResult(
            status="answered",
            intent=case.expected_intent,
            answer=answer,
            citations=validate_citations(answer, [evidence]),
        )
        rows.append(
            {
                "case_id": case.id,
                "expected_status": case.expected_status,
                "result": result.model_dump(mode="json"),
                "cited_evidence": [evidence.model_dump(mode="json")],
            }
        )
    return dataset, {
        "run_id": "frozen-run",
        "created_at": "2026-11-27T00:00:00Z",
        "model": "answer-model",
        "mode": "bm25_only",
        "dataset_sha256": dataset.sha256,
        "index_version": "test-index",
        "rows": rows,
    }


@pytest.fixture
def judge_settings() -> LLMSettings:
    return LLMSettings(
        provider="openai_compatible",
        model="judge-model",
        base_url="https://judge.example.test:8443/v1",
        api_key=SECRET,
        temperature=0.2,
        json_mode=True,
        max_retries=0,
    )


def _write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_single_arm_scores_a_selected_answer_and_audits_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_data: tuple[EvalDataset, dict],
    judge_settings: LLMSettings,
) -> None:
    dataset, report = fixture_data
    answers_path = _write_report(tmp_path / "frozen.json", report)
    output_path = tmp_path / "new" / "reports" / "out.json"
    case = dataset.cases[1]
    closed = []

    def respond(messages: Sequence[ChatMessage]) -> str:
        payload = json.loads(messages[1].content)
        assert payload["query"] == case.request.query
        assert payload["request"] == case.request.model_dump(mode="json", exclude={"query"})
        assert payload["cited_evidence"] == report["rows"][1]["cited_evidence"]
        return '{"faithfulness":0.75}'

    class Judge(MockLLMClient):
        provider = "openai_compatible"

        def close(self) -> None:
            closed.append(self.model)

    def create_client(config: LLMSettings) -> Judge:
        assert output_path.parent.is_dir()
        return Judge(config.model, responder=respond)

    monkeypatch.setattr("src.evaluation.replay.create_llm_client", create_client)
    result = replay_judge(
        judge_settings,
        answers_path=answers_path,
        dataset=dataset,
        output_path=output_path,
        case_ids={case.id},
    )
    assert closed == [judge_settings.model]
    assert result == json.loads(output_path.read_text(encoding="utf-8"))
    assert SECRET not in output_path.read_text(encoding="utf-8")
    assert result["run_id"] and result["created_at"].endswith("+00:00")
    assert result["judge_provider"] == judge_settings.provider
    assert result["judge_model"] == judge_settings.model
    assert result["judge_gateway"] == "https://judge.example.test:8443/v1"
    assert result["temperature"] == 0.2 and result["json_mode"] is True
    assert result["judge_prompt_sha256"] == judge_prompt_sha256()
    assert result["answer_run_id"] == report["run_id"] and result["answer_model"] == report["model"]
    assert result["answer_mode"] == report["mode"]
    assert result["answer_report_sha256"] == sha256(answers_path.read_bytes()).hexdigest()
    assert result["dataset_sha256"] == dataset.sha256
    assert result["index_version"] == report["index_version"]
    assert result["summary"] == {
        "with_request": {"attempted": 1, "scored": 1, "failed": 0, "mean": 0.75},
    }
    assert "paired" not in result and [row["case_id"] for row in result["rows"]] == [case.id]


def test_paired_legacy_smoke_and_phase_b_handle_invalid_scores_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_data: tuple[EvalDataset, dict],
    judge_settings: LLMSettings,
) -> None:
    dataset, report = fixture_data
    flat_rows = []
    for row in report["rows"]:
        result = row["result"]
        evidence = row.pop("cited_evidence")
        flat_rows.append(
            {
                **result,
                "case_id": row["case_id"],
                "expected_status": row["expected_status"],
                "cited_evidence": evidence,
            }
        )
        del flat_rows[-1]["citations"]
    flat_report = {**report, "summary": {"mode": "hybrid_rrf"}, "rows": flat_rows}
    del flat_report["mode"]
    legacy = _write_report(tmp_path / "frozen_smoke.json", report)
    flat = _write_report(tmp_path / "phase_b_v4_hybrid_rrf.json", flat_report)

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["cited_evidence"][0]["text"]
        with_request = "request" in payload
        if payload["query"] == dataset.cases[0].request.query:
            score = 0.75 if with_request else 0.25
        elif with_request:
            raise httpx.ReadTimeout(f"Simulated timeout with {SECRET}", request=request)
        else:
            score = 1.25
        message = {"content": json.dumps({"faithfulness": score})}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(
            "src.evaluation.replay.create_llm_client",
            lambda cfg: OpenAICompatibleLLMClient(cfg, http_client=http_client),
        )
        for answers_path, evidence_paths in [(legacy, (flat,)), (flat, ())]:
            output_path = tmp_path / f"{answers_path.stem}_replayed.json"
            result = replay_judge(
                judge_settings,
                answers_path=answers_path,
                dataset=dataset,
                output_path=output_path,
                evidence_paths=evidence_paths,
                paired=True,
            )
            assert result["paired"] == {"scored": 1, "mean_delta": 0.5}
            assert result["summary"] == {
                "without_request": {"attempted": 2, "scored": 1, "failed": 1, "mean": 0.25},
                "with_request": {"attempted": 2, "scored": 1, "failed": 1, "mean": 0.75},
            }
            assert result["rows"][1]["paired_delta"] is None
            assert all(arm["error"] for arm in result["rows"][1]["arms"].values())
            assert SECRET not in output_path.read_text(encoding="utf-8")
            assert result["evidence_report_sha256"] == [
                sha256(path.read_bytes()).hexdigest() for path in evidence_paths
            ]


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("hash", "SHA-256 differ"),
        ("order", "IDs or order differ"),
        ("status", "expected statuses differ"),
        ("chunk", "missing frozen chunk"),
    ],
)
def test_rejects_invalid_frozen_input_before_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_data: tuple[EvalDataset, dict],
    judge_settings: LLMSettings,
    invalid: str,
    message: str,
) -> None:
    dataset, report = fixture_data
    if invalid == "hash":
        report["dataset_sha256"] = "incorrect"
    if invalid == "order":
        report["rows"].reverse()
    if invalid == "status":
        report["rows"][1]["expected_status"] = "insufficient_evidence"
    if invalid == "chunk":
        report["rows"][0]["cited_evidence"] = []
    _assert_not_connected(monkeypatch)
    output = tmp_path / "out.json"
    with pytest.raises(CorpusError, match=message):
        replay_judge(
            judge_settings,
            answers_path=_write_report(tmp_path / "frozen.json", report),
            dataset=dataset,
            output_path=output,
            case_ids={dataset.cases[0].id},
        )
    assert not output.exists()


def _assert_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_client(_: LLMSettings) -> MockLLMClient:
        pytest.fail("Invalid input reached client creation")

    monkeypatch.setattr("src.evaluation.replay.create_llm_client", unexpected_client)


@pytest.mark.parametrize(
    "suffix", [f"{SECRET}@", f"user:{SECRET}@", f"?token={SECRET}", f"#{SECRET}"]
)
def test_rejects_unsafe_gateway_without_echoing_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    _assert_not_connected(monkeypatch)
    dataset = load_dataset(ROOT / "data" / "eval" / "golden_smoke.jsonl")
    gateway = (
        f"https://{suffix}judge.example.test/v1"
        if suffix.endswith("@")
        else f"https://judge.example.test/v1{suffix}"
    )
    settings = LLMSettings(provider="openai_compatible", model="judge-model", base_url=gateway)
    with pytest.raises(ConfigurationError, match="gateway URL must omit") as failure:
        replay_judge(
            settings,
            answers_path=tmp_path / "missing.json",
            dataset=dataset,
            output_path=tmp_path / "out.json",
        )
    assert SECRET not in str(failure.value)


@pytest.mark.parametrize("existing_output", ["file", "directory"])
def test_same_model_override_never_overwrites_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_data: tuple[EvalDataset, dict],
    judge_settings: LLMSettings,
    existing_output: str,
) -> None:
    dataset, report = fixture_data
    judge_settings.model = report["model"].upper()
    args = {
        "answers_path": _write_report(tmp_path / "frozen.json", report),
        "dataset": dataset,
        "output_path": tmp_path / "out.json",
    }
    _assert_not_connected(monkeypatch)
    with pytest.raises(ConfigurationError, match="must differ"):
        replay_judge(judge_settings, **args)
    output_path = args["output_path"]
    preserved = output_path
    if existing_output == "directory":
        output_path.mkdir()
        preserved = output_path / "data.txt"
    preserved.write_text("preserve this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        replay_judge(judge_settings, **args, allow_same_model=True)
    assert preserved.read_text(encoding="utf-8") == "preserve this"


@pytest.mark.parametrize("use_config", [True, False])
def test_replay_cli_accepts_both_configuration_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fixture_data: tuple[EvalDataset, dict],
    use_config: bool,
) -> None:
    dataset, report = fixture_data
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        "\n".join(case.model_dump_json() for case in dataset.cases) + "\n", encoding="utf-8"
    )
    report["dataset_sha256"] = load_dataset(dataset_path).sha256
    answers_path = _write_report(tmp_path / "frozen.json", report)
    output_path = tmp_path / "out.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DIAGAGENT_JUDGE_LLM__API_KEY", SECRET)
    cli_main = runpy.run_path(str(ROOT / "scripts" / "replay_judge.py"))["main"]
    expected_model = "profile-judge" if use_config else report["model"].upper()
    gateway = "https://judge.example.test/v1"

    def create_client(config: LLMSettings) -> MockLLMClient:
        assert config.model == expected_model and str(config.base_url) == gateway
        assert config.api_key is not None and config.api_key.get_secret_value() == SECRET
        return MockLLMClient(config.model, responder=lambda _: '{"faithfulness":0.8}')

    monkeypatch.setattr("src.evaluation.replay.create_llm_client", create_client)
    if use_config:
        profile = tmp_path / "judge.yaml"
        default = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        profile.write_text(
            default + "\njudge_llm:\n  provider: openai_compatible\n"
            f"  model: {expected_model}\n  base_url: {gateway}\n",
            encoding="utf-8",
        )
        flags = ["--config", str(profile)]
    else:
        flags = ["--model", expected_model, "--base-url", gateway, "--allow-same-model"]
    assert (
        cli_main(
            [
                "--answers",
                str(answers_path),
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_path),
                *flags,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert json.loads(captured.out)["with_request"]["scored"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["judge_model"] == expected_model
