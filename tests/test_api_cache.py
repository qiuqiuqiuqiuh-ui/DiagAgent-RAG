from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.agents.schemas import Citation, DiagnosticRequest, DiagnosticResult
from src.agents.tools import DiagnosticTools
from src.api.app import create_app
from src.api.cache import CacheScope, SQLiteSemanticCache, context_hash, profile_hash
from src.core.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    settings = load_settings(project_root=ROOT, env_file=None)
    settings.paths.index_dir = tmp_path / "index"
    settings.paths.cache_file = tmp_path / "cache.sqlite3"
    return settings


def test_semantic_cache_reuses_similar_vectors_and_isolates_scope(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    cache = SQLiteSemanticCache(db)
    scope = CacheScope("index-a", "profile-a", "context-a")
    citation = Citation(
        chunk_id="motor.md#s0#c0", doc_id="motor.md", source_path="motor.md", section_title="Motor"
    )
    result = DiagnosticResult(
        status="answered",
        intent="fault_diagnosis",
        answer="A motor issue. [motor.md#s0#c0]",
        citations=(citation,),
    )
    vector = np.array([1.0, 0.0], np.float32)
    cache.store(scope, "first wording", vector, result)
    reused = SQLiteSemanticCache(db).lookup(scope, np.array([0.96, 0.28], np.float32))
    assert reused == result
    assert cache.lookup(scope, np.array([0.94, 0.34], np.float32)) is None
    assert cache.lookup(scope, np.zeros(2, np.float32)) is None
    assert cache.lookup(scope, np.ones(3, np.float32)) is None
    assert cache.lookup(CacheScope("index-b", "profile-a", "context-a"), vector) is None
    assert cache.lookup(CacheScope("index-a", "profile-b", "context-a"), vector) is None
    assert cache.lookup(CacheScope("index-a", "profile-a", "context-b"), vector) is None


def test_context_hash_keeps_precision_and_entities(api_settings: Settings) -> None:
    tools = DiagnosticTools.from_corpus(api_settings.paths.corpus_dir)
    reference = DiagnosticRequest(
        query="What is the winding_temp threshold?",
        equipment_id=" MX-200 ",
        sensor_readings={"WINDING_TEMP": 119},
    )
    normalized = DiagnosticRequest(
        query="What is the winding_temp threshold?",
        equipment_id="mx-200",
        sensor_readings={"winding_temp": 119.0},
    )
    value = context_hash(reference, tools)
    assert value == context_hash(normalized, tools)
    assert value != context_hash(
        normalized.model_copy(update={"sensor_readings": {"winding_temp": 119.000001}}), tools
    )
    assert value != context_hash(normalized.model_copy(update={"equipment_id": "PM-14"}), tools)
    assert context_hash(DiagnosticRequest(query="Explain VFD-OC3"), tools) != context_hash(
        DiagnosticRequest(query="Explain VFD-OC9"), tools
    )
    assert context_hash(
        DiagnosticRequest(query="Check pressure above 1.8 bar"), tools
    ) != context_hash(DiagnosticRequest(query="Check pressure below 1.8 bar"), tools)
    assert context_hash(
        DiagnosticRequest(query="Check temperature at -10 degC"), tools
    ) != context_hash(DiagnosticRequest(query="Check temperature at 10 degC"), tools)
    original = profile_hash(api_settings)
    api_settings.llm.model = "other-llm"
    assert original != profile_hash(api_settings)


def test_api_ingestion_cold_cached_and_validation(api_settings: Settings) -> None:
    query = {"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}
    with TestClient(create_app(api_settings)) as client:
        assert client.get("/health").json()["status"] == "not_ready"
        assert client.post("/v1/diagnose", json=query).status_code == 503
        manifest = client.post("/v1/ingest")
        assert manifest.status_code == 200, manifest.text
        assert manifest.json()["chunk_count"] > 0
        assert client.get("/health").json()["index_version"] == manifest.json()["index_version"]
        assert client.post("/v1/diagnose", json={"query": " "}).status_code == 422
        assert (
            client.post(
                "/v1/diagnose", json={**query, "sensor_readings": {"temp": True}}
            ).status_code
            == 422
        )
        cold = client.post("/v1/diagnose", json=query).json()
        assert cold["status"] == "answered" and not cold["cache_hit"]
        assert cold["trace"]["retry_count"] == 0
        assert cold["citations"] and cold["tool_findings"][0]["status"] == "found"
        cached = client.post(
            "/v1/diagnose", json={**query, "query": "WHAT DOES VFD-OC3 MEAN?"}
        ).json()
        assert cached["cache_hit"] and cached["answer"] == cold["answer"] and "trace" not in cached
        bypassed = client.post(
            "/v1/diagnose", json={**query, "bypass_cache": True, "include_trace": False}
        ).json()
        assert (
            bypassed["status"] == "answered"
            and not bypassed["cache_hit"]
            and "trace" not in bypassed
        )

    with TestClient(create_app(api_settings)) as restarted:
        assert restarted.get("/health").json()["status"] == "ok"
        assert restarted.post("/v1/diagnose", json=query).json()["cache_hit"]


def test_api_cache_isolation_between_assets_readings_and_abstentions(
    api_settings: Settings,
) -> None:
    query = {
        "query": "What is the winding_temp warning threshold?",
        "sensor_readings": {"winding_temp": 118},
    }
    with TestClient(create_app(api_settings)) as client:
        client.post("/v1/ingest").raise_for_status()
        mx = client.post("/v1/diagnose", json={**query, "equipment_id": "MX-200"}).json()
        pm = client.post("/v1/diagnose", json={**query, "equipment_id": "PM-14"}).json()
        next_reading = client.post(
            "/v1/diagnose",
            json={**query, "equipment_id": "MX-200", "sensor_readings": {"winding_temp": 121}},
        ).json()
        assert [case["status"] for case in (mx, pm, next_reading)] == ["answered"] * 3
        assert [case["cache_hit"] for case in (mx, pm, next_reading)] == [False] * 3
        assert [case["tool_findings"][0]["status"] for case in (mx, pm, next_reading)] == [
            "warning",
            "critical",
            "critical",
        ]

        for _ in range(2):
            ambiguous = client.post("/v1/diagnose", json=query).json()
            assert ambiguous["status"] == "insufficient_evidence" and not ambiguous["cache_hit"]
            assert ambiguous["trace"]["retry_count"] == 0


def test_reingest_preserves_active_snapshot_on_invalid_rules_and_rotates_cache(
    api_settings: Settings, tmp_path: Path
) -> None:
    api_settings.paths.corpus_dir = copytree(api_settings.paths.corpus_dir, tmp_path / "corpus")
    query = {"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}
    with TestClient(create_app(api_settings)) as client:
        old_version = client.post("/v1/ingest").json()["index_version"]
        assert not client.post("/v1/diagnose", json=query).json()["cache_hit"]
        thresholds = api_settings.paths.corpus_dir / "sensor_thresholds.json"
        original = thresholds.read_text(encoding="utf-8")
        thresholds.write_text(
            original.replace('"warning_threshold": 105', '"warning_threshold": true'),
            encoding="utf-8",
        )
        assert client.post("/v1/ingest").status_code == 422
        assert client.get("/health").json()["index_version"] == old_version
        assert client.post("/v1/diagnose", json=query).json()["cache_hit"]
        thresholds.write_text(original, encoding="utf-8")
        manual = api_settings.paths.corpus_dir / "manuals" / "vfd_faults.md"
        with manual.open("a", encoding="utf-8") as stream:
            stream.write("\n\nRecord a follow-up after the maintenance action.\n")
        new_version = client.post("/v1/ingest").json()["index_version"]
        assert new_version != old_version
        assert not client.post("/v1/diagnose", json=query).json()["cache_hit"]
