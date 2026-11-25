import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import Response

from src.api.app import create_app
from src.api.schemas import DiagnoseResponse, HealthResponse
from src.core.exceptions import DiagAgentError
from src.core.logging import configure_logging
from src.core.settings import Settings, load_settings
from src.evaluation.dataset import load_dataset
from src.evaluation.report import write_reports
from src.evaluation.runner import evaluate
from src.ingestion.corpus import read_corpus
from src.retrieval.index import IndexManifest, load_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _success(response: Response) -> object:
    if not response.is_success:
        raise RuntimeError(
            f"{response.request.method} {response.request.url.path} "
            f"returned HTTP {response.status_code}"
        )
    return response.json()


def _verify_api(settings: Settings) -> IndexManifest:
    with TestClient(create_app(settings)) as client:
        before = HealthResponse.model_validate(_success(client.get("/health")))
        _require(before.status == "not_ready", "Fresh service should wait for ingestion")
        _require(
            client.post("/v1/diagnose", json={"query": "What does VFD-OC3 mean?"}).status_code
            == 503,
            "Diagnosis before ingestion should return HTTP 503",
        )

        manifest = IndexManifest.model_validate(_success(client.post("/v1/ingest")))
        corpus = read_corpus(settings.paths.corpus_dir, settings.ingestion)
        stored = load_index(
            settings.paths.index_dir,
            settings.embedding,
            settings.ingestion,
            expected_corpus_hash=corpus.corpus_hash,
        )
        _require(stored.manifest == manifest, "Published manifest must reload and match the API")
        health = HealthResponse.model_validate(_success(client.get("/health")))
        _require(
            health.status == "ok" and health.index_version == manifest.index_version,
            "Health should report the loaded index",
        )
        print(
            f"PASS ingest and manifest: "
            f"{corpus.document_count} documents, {manifest.chunk_count} chunks"
        )
        print("PASS /health: not_ready before ingestion and ok afterward")

        payload = {"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}
        cold = DiagnoseResponse.model_validate(_success(client.post("/v1/diagnose", json=payload)))
        _require(
            cold.status == "answered"
            and not cold.cache_hit
            and bool(cold.citations)
            and cold.trace is not None,
            "Cold diagnosis should answer with citations and a trace",
        )
        cached = DiagnoseResponse.model_validate(
            _success(client.post("/v1/diagnose", json=payload))
        )
        _require(
            cached.cache_hit and cached.answer == cold.answer and cached.trace is None,
            "Cache hit should reuse the grounded answer",
        )
        bypassed = DiagnoseResponse.model_validate(
            _success(
                client.post(
                    "/v1/diagnose", json={**payload, "bypass_cache": True, "include_trace": False}
                )
            )
        )
        _require(
            not bypassed.cache_hit and bypassed.status == "answered" and bypassed.trace is None,
            "Bypass should run the workflow without a trace",
        )

        abstention = DiagnoseResponse.model_validate(
            _success(
                client.post(
                    "/v1/diagnose",
                    json={"query": "What is the mounting bolt torque for the MX-200 motor?"},
                )
            )
        )
        _require(
            abstention.status == "insufficient_evidence"
            and not abstention.cache_hit
            and not abstention.citations
            and bool(abstention.missing_aspects)
            and abstention.trace is not None
            and abstention.trace.retry_count == 2,
            "Missing documentation should cause abstention after two retries",
        )
        refusal = DiagnoseResponse.model_validate(
            _success(client.post("/v1/diagnose", json={"query": "What is the weather forecast?"}))
        )
        _require(
            refusal.status == "out_of_scope"
            and refusal.trace is not None
            and len(refusal.trace.nodes) == 1,
            "Out-of-scope input should stop at the router",
        )
        print("PASS /v1/diagnose: cold, cached, bypass, abstention, and out-of-scope")
        return manifest


def main() -> int:
    try:
        with patch.dict(os.environ), TemporaryDirectory(prefix="diagagent-verify-") as directory:
            for key in list(os.environ):
                if key.upper().startswith(("DIAGAGENT_", "LANGSMITH_", "LANGCHAIN_")):
                    del os.environ[key]
            settings = load_settings(project_root=PROJECT_ROOT, env_file=None)
            providers = {
                settings.embedding.provider,
                settings.llm.provider,
                settings.reranker.provider,
            }
            _require(
                providers == {"mock"}, "The shipped offline verifier requires all Mock providers"
            )
            temporary = Path(directory)
            settings.paths.index_dir = temporary / "index"
            settings.paths.cache_file = temporary / "semantic.sqlite3"
            settings.paths.reports_dir = temporary / "reports"
            settings.log_level = "WARNING"
            configure_logging(settings.log_level)

            manifest = _verify_api(settings)
            dataset = load_dataset(PROJECT_ROOT / "data" / "eval" / "golden_smoke.jsonl")
            report = evaluate(settings, dataset)
            json_path, markdown_path = write_reports(report, settings.paths.reports_dir)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            _require(
                report.run_mode == "mock"
                and report.index_version == manifest.index_version
                and len(report.modes) == 4
                and all(mode.metrics.samples == 15 for mode in report.modes)
                and saved["dataset_sha256"] == dataset.sha256
                and markdown_path.stat().st_size > 0,
                "Smoke evaluation and both report formats must match the indexed dataset",
            )
            reranked = next(mode for mode in report.modes if mode.mode == "hybrid_rerank")
            _require(
                all(case.status == case.expected_status for case in reranked.cases),
                "The default workflow must match the labeled smoke statuses",
            )
            print("PASS Mock smoke evaluation: 15 cases x 4 retrieval modes")
    except (DiagAgentError, OSError, RuntimeError) as exc:
        print(f"FAIL offline verification: {exc}", file=sys.stderr)
        return 1
    print("PASS offline end-to-end verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
