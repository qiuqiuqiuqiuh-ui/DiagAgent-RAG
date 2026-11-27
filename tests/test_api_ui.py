from pathlib import Path
from shutil import copytree
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from pydantic import AnyHttpUrl, SecretStr

from src.api.app import create_app
from src.core.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
CHUNK_ID = "manuals/vfd_faults.md#s1#c0"


@pytest.fixture
def demo_settings(tmp_path: Path) -> Settings:
    settings = load_settings(project_root=ROOT, env_file=None)
    settings.paths.corpus_dir = copytree(settings.paths.corpus_dir, tmp_path / "corpus")
    settings.paths.index_dir = tmp_path / "index"
    settings.paths.cache_file = tmp_path / "cache.sqlite3"
    return settings


@pytest.mark.parametrize("mode", ["mock", "live"])
def test_demo_meta_exposes_only_model_names_and_active_rules(
    demo_settings: Settings, mode: str
) -> None:
    if mode == "live":
        demo_settings.llm.provider = "openai_compatible"
    demo_settings.llm.api_key = SecretStr("test-llm-secret")
    demo_settings.embedding.api_key = SecretStr("test-embedding-secret")
    demo_settings.llm.base_url = AnyHttpUrl("https://user:test-url-secret@gateway.test/v1")
    with TestClient(create_app(demo_settings)) as client:
        before = client.get("/v1/meta")
        assert before.status_code == 200
        assert before.json() == {
            "domain": demo_settings.domain.model_dump(),
            "mode": mode,
            "llm_model": demo_settings.llm.model,
            "embedding_model": demo_settings.embedding.model,
            "reranker_model": demo_settings.reranker.model,
            "equipment_ids": [],
            "fault_codes": {},
            "sensors": [],
        }
        assert client.post("/v1/ingest").status_code == 200
        after = client.get("/v1/meta")

    assert after.status_code == 200
    assert "secret" not in before.text + after.text
    assert "gateway.test" not in before.text + after.text
    meta = after.json()
    assert "DRV-300" in meta["equipment_ids"]
    assert meta["fault_codes"]["DRV-300"] == ["E01.2", "VFD-OC3", "VFD-UV1"]
    assert {
        "equipment_id": "PM-14",
        "sensor": "winding_temp",
        "unit": "degC",
        "warning_threshold": 100,
        "critical_threshold": 115,
        "direction": "high",
    } in meta["sensors"]


def test_demo_chunk_uses_only_the_latest_active_snapshot(demo_settings: Settings) -> None:
    chunk_url = "/v1/chunks/" + quote(CHUNK_ID, safe="/")
    with TestClient(create_app(demo_settings)) as client:
        assert client.get(chunk_url).status_code == 503
        first = client.post("/v1/ingest")
        assert first.status_code == 200
        existing = client.get(chunk_url)
        assert existing.status_code == 200
        assert existing.json() == {
            "chunk_id": CHUNK_ID,
            "doc_id": "manuals/vfd_faults.md",
            "source_path": "manuals/vfd_faults.md",
            "section_title": existing.json()["section_title"],
            "page_number": None,
            "text": existing.json()["text"],
        }
        assert "VFD-OC3" in existing.json()["section_title"]
        assert "overcurrent" in existing.json()["text"].lower()
        assert client.get("/v1/chunks/unknown%23s0%23c0").status_code == 404
        assert client.get(f"/v1/chunks/%2e%2e%2f{quote(CHUNK_ID, safe='')}").status_code == 404

        old_manual = demo_settings.paths.corpus_dir / "manuals" / "vfd_faults.md"
        old_manual.rename(old_manual.with_name("drive_faults.md"))
        assert client.get(chunk_url).json() == existing.json()
        second = client.post("/v1/ingest")
        assert second.status_code == 200
        assert first.json()["index_version"] != second.json()["index_version"]
        assert client.get(chunk_url).status_code == 404
        renamed = CHUNK_ID.replace("vfd_faults", "drive_faults")
        assert client.get("/v1/chunks/" + quote(renamed, safe="")).json()["chunk_id"] == renamed


def test_demo_static_resources_and_diagnosis_to_source(demo_settings: Settings) -> None:
    with TestClient(create_app(demo_settings)) as client:
        for path, content_type in [
            ("/", "text/html"),
            ("/static/app.js", "javascript"),
            ("/static/styles.css", "text/css"),
        ]:
            response = client.get(path)
            assert response.status_code == 200
            assert content_type in response.headers["content-type"]
        assert client.post("/v1/ingest").status_code == 200
        diagnosis = client.post(
            "/v1/diagnose",
            json={
                "query": "What does VFD-OC3 mean?",
                "equipment_id": "DRV-300",
                "fault_code": "VFD-OC3",
            },
        )
        assert diagnosis.status_code == 200
        result = diagnosis.json()
        assert result["status"] == "answered" and result["citations"]
        for citation in result["citations"]:
            assert f"[{citation['chunk_id']}]" in result["answer"]
            quoted_id = quote(citation["chunk_id"], safe="")
            chunk = client.get(f"/v1/chunks/{quoted_id}")
            assert chunk.status_code == 200 and chunk.json()["text"]
            assert chunk.json()["source_path"] == citation["source_path"]
