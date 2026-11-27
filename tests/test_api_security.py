from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.app import create_app
from src.core.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
QUERY = {"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}
USER = {"X-API-Key": "test-diagnostic-credential"}
ADMIN = {"Authorization": "bEaReR test-administrator-credential"}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = load_settings(project_root=ROOT, env_file=None)
    config.paths.index_dir = tmp_path / "index"
    config.paths.cache_file = tmp_path / "cache.sqlite3"
    return config


def test_default_allows_anonymous_local_api(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/meta").status_code == 200
        assert client.post("/v1/ingest").status_code == 200
        assert client.post("/v1/diagnose", json=QUERY).status_code == 200


def test_api_and_admin_keys_are_applied_before_body_validation(settings: Settings) -> None:
    settings.api.api_key = SecretStr(USER["X-API-Key"])
    settings.api.admin_api_key = SecretStr(ADMIN["Authorization"].split()[1])
    evidence = "/v1/chunks/manuals/vfd_faults.md%23s1%23c0"
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/meta").status_code == 401
        assert client.get(evidence, headers={"X-API-Key": "invalid"}).status_code == 401
        invalid = client.post(
            "/v1/diagnose", content=b"{", headers={"Content-Type": "application/json"}
        )
        assert invalid.status_code == 401 and invalid.headers["WWW-Authenticate"] == "Bearer"
        for response in [invalid, client.post("/v1/ingest", headers=USER)]:
            assert "credential" in response.json()["detail"].lower()
            assert USER["X-API-Key"] not in response.text
            assert ADMIN["Authorization"].split()[1] not in response.text
        assert client.post("/v1/ingest", headers=USER).status_code == 403
        assert client.post("/v1/ingest").status_code == 401
        assert client.post("/v1/ingest", headers=ADMIN).status_code == 200
        assert client.get("/v1/meta", headers=USER).status_code == 200
        assert client.get(evidence, headers=USER).status_code == 200
        bearer = {"Authorization": f"Bearer {USER['X-API-Key']}"}
        assert client.post("/v1/diagnose", headers=bearer, json=QUERY).status_code == 200


def test_admin_ingest_falls_back_to_regular_key(settings: Settings) -> None:
    settings.api.api_key = SecretStr(USER["X-API-Key"])
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/ingest").status_code == 401
        assert client.post("/v1/ingest", headers=USER).status_code == 200


def test_max_query_and_sensor_counts(settings: Settings) -> None:
    settings.api.max_query_chars = 24
    settings.api.max_sensor_readings = 1
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/ingest").status_code == 200
        assert (
            client.post(
                "/v1/diagnose", json={"query": "x" * 24, "sensor_readings": {"temp": 1}}
            ).status_code
            == 200
        )
        assert client.post("/v1/diagnose", json={"query": "温" * 25}).status_code == 422
        assert (
            client.post(
                "/v1/diagnose", json={"query": "VFD-OC3", "sensor_readings": {"a": 1, "b": 2}}
            ).status_code
            == 422
        )


def test_token_bucket_refills_and_probes_are_unlimited(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    monkeypatch.setattr("src.api.security.monotonic", lambda: clock[0])
    settings.api.rate_limit_rpm = 2
    settings.api.api_key = SecretStr(USER["X-API-Key"])
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/meta").status_code == 401
        assert [client.get("/v1/meta", headers=USER).status_code for _ in range(2)] == [200, 200]
        blocked = client.get("/v1/meta", headers={**USER, "X-Forwarded-For": "192.0.2.9"})
        assert blocked.status_code == 429 and blocked.headers["Retry-After"] == "30"
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/ingest", headers=USER).status_code == 200
        clock[0] += 30
        assert client.get("/v1/meta", headers=USER).status_code == 200


def test_cors_preflight_requires_an_allowed_origin_without_api_key(settings: Settings) -> None:
    origin = "https://ops.example.test"
    settings.api.cors_allow_origins = (origin,)
    settings.api.api_key = SecretStr(USER["X-API-Key"])
    with TestClient(create_app(settings)) as client:
        preflight = {
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, X-Request-ID",
        }
        allowed = client.options("/v1/diagnose", headers={**preflight, "Origin": origin})
        assert (
            allowed.status_code == 200 and allowed.headers["Access-Control-Allow-Origin"] == origin
        )
        refused = client.options(
            "/v1/diagnose", headers={**preflight, "Origin": "https://elsewhere.test"}
        )
        assert refused.status_code == 400 and "Access-Control-Allow-Origin" not in refused.headers
