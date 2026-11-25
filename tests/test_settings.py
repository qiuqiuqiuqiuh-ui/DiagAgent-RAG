from pathlib import Path

import pytest

from src.core.exceptions import ConfigurationError
from src.core.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_and_colab_profiles() -> None:
    default = load_settings(project_root=PROJECT_ROOT, env_file=None)
    assert default.domain.language == "en"
    assert default.paths.corpus_dir == PROJECT_ROOT / "data" / "raw"
    assert default.ingestion.chunk_overlap < default.ingestion.chunk_size
    assert (default.llm.provider, default.embedding.provider, default.reranker.provider) == (
        "mock",
        "mock",
        "mock",
    )

    colab = load_settings("configs/colab.yaml", project_root=PROJECT_ROOT, env_file=None)
    assert colab.paths.index_dir.as_posix().endswith(
        "/content/drive/MyDrive/DiagAgent-RAG/data/index"
    )
    assert colab.llm.provider == "openai_compatible"
    assert colab.llm.api_key is None
    assert colab.embedding.dimension == 384
    assert colab.embedding.device == colab.reranker.device == "cuda"


def test_environment_overrides_dotenv_and_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "local.env").write_text(
        "DIAGAGENT_EMBEDDING__DIMENSION=16\n"
        "DIAGAGENT_PATHS__CORPUS_DIR=my-corpus\n"
        "DIAGAGENT_LLM__API_KEY=example-test-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DIAGAGENT_EMBEDDING__DIMENSION", "64")
    profile = PROJECT_ROOT / "configs" / "default.yaml"
    settings = load_settings(profile, project_root=tmp_path, env_file="local.env")
    assert settings.embedding.dimension == 64
    assert settings.embedding.model == "mock-hash-v1"
    assert settings.paths.corpus_dir == tmp_path / "my-corpus"
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "example-test-key"
    assert "example-test-key" not in repr(settings)
    assert load_settings(profile, project_root=tmp_path, env_file=None).llm.api_key is None


def test_rejects_invalid_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIAGAGENT_INGESTION__CHUNK_OVERLAP", "1000")
    with pytest.raises(ConfigurationError, match="ingestion"):
        load_settings(project_root=PROJECT_ROOT, env_file=None)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("[", "Could not read YAML profile"),
        ("- domain", "must be a mapping"),
        ("_env_file: other.env", "reserved setting name"),
    ],
)
def test_rejects_invalid_profiles(tmp_path: Path, contents: str, error: str) -> None:
    (tmp_path / "config.yaml").write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=error):
        load_settings("config.yaml", project_root=tmp_path, env_file=None)


def test_rejects_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Could not read YAML profile"):
        load_settings(project_root=tmp_path, env_file=None)


@pytest.mark.parametrize("value", ["", " \t "])
@pytest.mark.parametrize("field", ["CORPUS_DIR", "INDEX_DIR", "REPORTS_DIR"])
def test_rejects_blank_path_overrides(
    monkeypatch: pytest.MonkeyPatch, value: str, field: str
) -> None:
    monkeypatch.setenv(f"DIAGAGENT_PATHS__{field}", value)
    with pytest.raises(ConfigurationError, match=rf"paths\.{field.lower()}"):
        load_settings(project_root=PROJECT_ROOT, env_file=None)


def test_rejects_invalid_dotenv_encoding(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DIAGAGENT_LOG_LEVEL=INFO", encoding="utf-16")
    with pytest.raises(ConfigurationError, match="Could not load settings"):
        load_settings(project_root=PROJECT_ROOT, env_file=env_path)


def test_normalizes_log_level_and_rejects_nonpositive_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIAGAGENT_LOG_LEVEL", " info ")
    assert load_settings(project_root=PROJECT_ROOT, env_file=None).log_level == "INFO"
    monkeypatch.setenv("DIAGAGENT_EMBEDDING__BATCH_SIZE", "0")
    with pytest.raises(ConfigurationError, match=r"embedding\.batch_size"):
        load_settings(project_root=PROJECT_ROOT, env_file=None)
