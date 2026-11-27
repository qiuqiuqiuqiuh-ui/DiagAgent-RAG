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


def test_manual_only_corpus_requires_explicit_rule_opt_out(demo_settings: Settings) -> None:
    for rule_path in demo_settings.rules.source_paths:
        (demo_settings.paths.corpus_dir / rule_path).unlink()
    with TestClient(create_app(demo_settings)) as strict:
        assert strict.post("/v1/ingest").status_code == 422
        assert strict.get("/health").json()["status"] == "not_ready"

    demo_settings.rules.allow_missing_rule_tables = True
    with TestClient(create_app(demo_settings)) as client:
        assert client.post("/v1/ingest").status_code == 200
        assert client.get("/v1/meta").json()["equipment_ids"] == []
        response = client.post(
            "/v1/diagnose", json={"query": "What does VFD-OC3 mean?", "equipment_id": "DRV-300"}
        )
        assert response.status_code == 200
        answer = response.json()
        assert answer["status"] == "answered"
        assert "overcurrent" in answer["answer"].lower()
        assert any(
            citation["source_path"] == "manuals/vfd_faults.md" for citation in answer["citations"]
        )


def test_chunk_lookup_handles_non_ascii_and_spaces_via_query_and_path(
    demo_settings: Settings,
) -> None:
    manual = demo_settings.paths.corpus_dir / "manuals" / "变频器 手册.md"
    manual.write_text("# 变频器手册\n\nVFD-CN9 加速过流说明。\n", encoding="utf-8")
    with TestClient(create_app(demo_settings)) as client:
        assert client.post("/v1/ingest").status_code == 200
        diagnosis = client.post(
            "/v1/diagnose",
            json={"query": "VFD-CN9 加速过流说明", "equipment_id": "DRV-300"},
        )
        assert diagnosis.status_code == 200
        result = diagnosis.json()
        assert result["status"] == "answered"
        target = next(
            citation["chunk_id"]
            for citation in result["citations"]
            if citation["source_path"] == "manuals/变频器 手册.md"
        )
        for url in (
            f"/v1/chunks?chunk_id={quote(target, safe='')}",
            f"/v1/chunks/{quote(target, safe='')}",
            f"/v1/chunks/{quote(target, safe='/')}",
        ):
            response = client.get(url)
            assert response.status_code == 200
            payload = response.json()
            assert payload["chunk_id"] == target
            assert payload["source_path"] == "manuals/变频器 手册.md"
            assert "VFD-CN9" in payload["text"]
        assert (
            client.get(f"/v1/chunks?chunk_id={quote(f'../{target}', safe='')}").status_code == 404
        )
        assert client.get(f"/v1/chunks/%2e%2e%2f{quote(target, safe='')}").status_code == 404


def test_demo_ui_assets_avoid_hardcoded_sample_equipment_and_support_custom_corpus(
    tmp_path: Path,
) -> None:
    settings = load_settings(project_root=ROOT, env_file=None)
    corpus = tmp_path / "custom_corpus"
    (corpus / "manuals").mkdir(parents=True)
    (corpus / "manuals" / "chiller.md").write_text(
        "# CHL-09 Chiller\n\nALM-99 indicates condenser high pressure.\n", encoding="utf-8"
    )
    (corpus / "fault_codes.csv").write_text(
        "fault_code,equipment_id,severity,meaning,first_check,escalation\n"
        "ALM-99,CHL-09,trip,Condenser high pressure,Inspect cooling tower,HVAC maintenance\n",
        encoding="utf-8",
    )
    (corpus / "sensor_thresholds.json").write_text(
        '[{"equipment_id":"CHL-09","sensor":"discharge_pressure","unit":"MPa",'
        '"warning_threshold":1.6,"critical_threshold":2.0,"direction":"high"}]',
        encoding="utf-8",
    )
    settings.paths.corpus_dir = corpus
    settings.paths.index_dir = tmp_path / "index"
    settings.paths.cache_file = tmp_path / "cache.sqlite3"
    with TestClient(create_app(settings)) as client:
        html = client.get("/").text
        script = client.get("/static/app.js").text
        for sample_token in ("DRV-300", "VFD-OC3", "PM-14", "MX-200"):
            assert sample_token not in html
            assert sample_token not in script
        assert client.post("/v1/ingest").status_code == 200
        meta = client.get("/v1/meta").json()
        assert meta["equipment_ids"] == ["CHL-09"]
        assert meta["fault_codes"] == {"CHL-09": ["ALM-99"]}
        assert meta["sensors"][0]["sensor"] == "discharge_pressure"


def test_demo_ui_toggle_and_authenticated_access(demo_settings: Settings) -> None:
    demo_settings.api.enable_demo_ui = False
    with TestClient(create_app(demo_settings)) as client:
        assert client.get("/").status_code == 404
        assert client.get("/static/app.js").status_code == 404
        assert client.get("/health").status_code == 200
        assert client.get("/v1/meta").status_code == 200

    demo_settings.api.enable_demo_ui = True
    demo_settings.api.api_key = SecretStr("ui-test-key")
    headers = {"X-API-Key": "ui-test-key"}
    with TestClient(create_app(demo_settings)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/v1/meta").status_code == 401
        assert client.get("/v1/meta", headers=headers).status_code == 200
        assert client.post("/v1/ingest", headers=headers).status_code == 200
        query_url = f"/v1/chunks?chunk_id={quote(CHUNK_ID, safe='')}"
        assert client.get(query_url).status_code == 401
        assert client.get(query_url, headers=headers).status_code == 200
