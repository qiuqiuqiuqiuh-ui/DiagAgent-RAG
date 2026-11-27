import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.app import create_app
from src.core.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = load_settings(project_root=ROOT, env_file=None)
    config.paths.index_dir = tmp_path / "index"
    config.paths.cache_file = tmp_path / "cache.sqlite3"
    return config


def test_request_id_propagates_to_response_and_worker_logs(
    settings: Settings, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging.getLogger("diagagent"), "propagate", True)
    caplog.set_level(logging.INFO, logger="diagagent")
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/v1/ingest", headers={"X-Request-ID": "rebuild-01"}).headers[
                "X-Request-ID"
            ]
            == "rebuild-01"
        )
        caplog.clear()
        first = client.post("/v1/diagnose", json=QUESTION, headers={"X-Request-ID": "diag-01"})
        assert first.status_code == 200 and first.headers["X-Request-ID"] == "diag-01"
        assert {record.event_fields["request_id"] for record in caplog.records} == {"diag-01"}
        assert "api.diagnose" in {record.message for record in caplog.records}
        caplog.clear()
        second = client.post("/v1/diagnose", json=QUESTION, headers={"X-Request-ID": "invalid id"})
        request_id = second.headers["X-Request-ID"]
        assert second.status_code == 200 and len(request_id) == 32
        assert {record.event_fields["request_id"] for record in caplog.records} == {request_id}


def test_health_probes_and_opt_in_metrics_cover_request_and_workflow_outcomes(
    settings: Settings,
) -> None:
    settings.api.enable_metrics = True
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert "diagagent_index_ready 0\n" in client.get("/metrics").text
        manifest = client.post("/v1/ingest").json()
        ready = client.get("/health/ready")
        assert (
            ready.status_code == 200 and ready.json()["index_version"] == manifest["index_version"]
        )

        assert not client.post("/v1/diagnose", json=QUESTION).json()["cache_hit"]
        assert client.post("/v1/diagnose", json=QUESTION).json()["cache_hit"]
        missing = client.post(
            "/v1/diagnose",
            json={
                "query": "What is the mounting bolt torque for the MX-200 motor?",
                "include_trace": False,
                "bypass_cache": True,
            },
        ).json()
        assert missing["status"] == "insufficient_evidence" and "trace" not in missing
        assert (
            client.post(
                "/v1/diagnose", json={"query": "What is the Tokyo weather forecast for tomorrow?"}
            ).json()["status"]
            == "out_of_scope"
        )
        assert client.post("/v1/diagnose", json={"query": ""}).status_code == 422
        client.get("/v1/chunks/manuals/vfd_faults.md%23s1%23c0").raise_for_status()
        assert client.get("/path-with-user-supplied-value").status_code == 404

        response = client.get("/metrics")
        assert response.status_code == 200 and response.headers["content-type"].startswith(
            "text/plain; version=0.0.4"
        )
        lines = set(response.text.splitlines())
        assert response.text.endswith("\n")
        assert {
            'diagagent_http_requests_total{endpoint="/health/ready",status="503"} 1',
            'diagagent_http_requests_total{endpoint="/health/ready",status="200"} 1',
            'diagagent_http_requests_total{endpoint="/v1/diagnose",status="200"} 4',
            'diagagent_http_requests_total{endpoint="/v1/diagnose",status="422"} 1',
            'diagagent_http_requests_total{endpoint="/v1/chunks/{chunk_id:path}",status="200"} 1',
            'diagagent_http_requests_total{endpoint="unmatched",status="404"} 1',
            'diagagent_diagnoses_total{status="answered"} 2',
            'diagagent_diagnoses_total{status="insufficient_evidence"} 1',
            'diagagent_diagnoses_total{status="out_of_scope"} 1',
            'diagagent_cache_requests_total{outcome="hit"} 1',
            'diagagent_cache_requests_total{outcome="miss"} 2',
            'diagagent_cache_requests_total{outcome="bypass"} 1',
            'diagagent_workflow_retries_total{retries="2"} 1',
            'diagagent_diagnose_duration_seconds_bucket{le="+Inf"} 4',
            "diagagent_diagnose_duration_seconds_count 4",
            "diagagent_index_ready 1",
            f"diagagent_index_chunks {manifest['chunk_count']}",
        } <= lines
        assert "path-with-user-supplied-value" not in response.text


def test_metrics_disabled_by_default_and_requires_admin_when_secured(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").status_code == 404
    settings.api.enable_metrics = True
    settings.api.api_key = SecretStr("ordinary-test-key")
    settings.api.admin_api_key = SecretStr("admin-test-key")
    with TestClient(create_app(settings)) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"X-API-Key": "ordinary-test-key"}).status_code == 403
        assert (
            client.get("/metrics", headers={"Authorization": "Bearer admin-test-key"}).status_code
            == 200
        )


def test_internal_errors_echo_request_id_with_generic_response(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/simulated-failure")
    def fail() -> None:
        raise RuntimeError("exception fixture")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/simulated-failure", headers={"X-Request-ID": "error-42"})
        assert response.status_code == 500 and response.headers["X-Request-ID"] == "error-42"
        assert "exception fixture" not in response.text
