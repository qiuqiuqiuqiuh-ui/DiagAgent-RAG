from pathlib import Path
from traceback import format_exception

import pytest
from pydantic import SecretStr

from src.api.cache import profile_hash
from src.core.exceptions import ConfigurationError
from src.core.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_and_colab_profiles() -> None:
    default = load_settings(project_root=PROJECT_ROOT, env_file=None)
    assert default.domain.language == "en"
    assert default.paths.corpus_dir == PROJECT_ROOT / "data" / "raw"
    assert default.ingestion.chunk_overlap < default.ingestion.chunk_size
    assert default.judge_llm is None
    assert (default.llm.provider, default.embedding.provider, default.reranker.provider) == (
        "mock",
        "mock",
        "mock",
    )

    colab = load_settings("configs/colab.yaml", project_root=PROJECT_ROOT, env_file=None)
    assert colab.paths.index_dir.as_posix().endswith(
        "/content/drive/MyDrive/DiagAgent-RAG/data/index"
    )
    assert colab.paths.cache_file.as_posix().endswith("/content/diagagent-cache/semantic.sqlite3")
    assert colab.llm.provider == "openai_compatible"
    assert colab.llm.api_key is None
    assert colab.judge_llm is None
    assert colab.embedding.dimension == 384
    assert colab.embedding.device == colab.reranker.device == "cuda"


def test_bilingual_production_template_loads_without_credentials() -> None:
    settings = load_settings(
        "configs/production_bilingual.yaml", project_root=PROJECT_ROOT, env_file=None
    )
    assert (settings.embedding.model, settings.embedding.dimension) == ("BAAI/bge-m3", 1024)
    assert settings.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert settings.domain.language == "zh"
    assert settings.judge_llm is not None and settings.judge_llm.model != settings.llm.model
    assert settings.judge_llm.api_key is settings.llm.api_key is None
    assert settings.rules.source_paths == ("fault_codes.csv", "sensor_thresholds.json")
    assert not settings.rules.allow_missing_rule_tables


def test_rule_paths_accept_nested_env_override_and_reject_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIAGAGENT_RULES__FAULT_CODES_PATH", "rules/codes.csv")
    monkeypatch.setenv("DIAGAGENT_RULES__ALLOW_MISSING_RULE_TABLES", "true")
    settings = load_settings(project_root=PROJECT_ROOT, env_file=None)
    assert settings.rules.source_paths == ("rules/codes.csv", "sensor_thresholds.json")
    assert settings.rules.allow_missing_rule_tables
    monkeypatch.setenv("DIAGAGENT_RULES__FAULT_CODES_PATH", "../codes.csv")
    with pytest.raises(ConfigurationError, match="rules"):
        load_settings(project_root=PROJECT_ROOT, env_file=None)


def test_judge_yaml_and_env_overrides_preserve_diagnostic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_profile = PROJECT_ROOT / "configs" / "default.yaml"
    baseline = load_settings(default_profile, env_file=None)
    profile = tmp_path / "judge.yaml"
    profile.write_text(
        default_profile.read_text(encoding="utf-8")
        + "\njudge_llm:\n  provider: openai_compatible\n  model: yaml-judge\n",
        encoding="utf-8",
    )
    configured = load_settings(profile, env_file=None)
    assert configured.judge_llm is not None
    assert configured.judge_llm.model == "yaml-judge"

    secret = "test-judge-secret"
    monkeypatch.setenv("DIAGAGENT_JUDGE_LLM__API_KEY", secret)
    with_key = load_settings(profile, env_file=None).judge_llm
    assert with_key is not None and with_key.model == "yaml-judge"
    assert isinstance(with_key.api_key, SecretStr) and with_key.api_key.get_secret_value() == secret

    overrides = {
        "PROVIDER": "openai_compatible",
        "MODEL": "env-judge",
        "BASE_URL": "https://judge.example.test/v1",
        "API_KEY": secret,
        "TEMPERATURE": "0.2",
        "JSON_MODE": "true",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(f"DIAGAGENT_JUDGE_LLM__{name}", value)
    from_yaml = load_settings(profile, env_file=None)
    from_env = load_settings(default_profile, env_file=None)
    judge = from_yaml.judge_llm
    assert judge is not None and judge == from_env.judge_llm
    assert judge.provider == "openai_compatible" and judge.model == "env-judge"
    assert str(judge.base_url) == overrides["BASE_URL"]
    assert judge.temperature == 0.2 and judge.json_mode is True
    assert isinstance(judge.api_key, SecretStr)
    assert judge.api_key.get_secret_value() == overrides["API_KEY"]
    assert overrides["API_KEY"] not in repr(from_yaml) + from_yaml.model_dump_json()
    assert profile_hash(baseline) == profile_hash(configured) == profile_hash(from_env)


@pytest.mark.parametrize("profile", ["default.yaml", "colab.yaml"])
@pytest.mark.parametrize("source", ["env", "dotenv"])
def test_standalone_judge_key_keeps_the_default_judge_unset(
    profile: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "DIAGAGENT_JUDGE_LLM__API_KEY"
    if source == "env":
        monkeypatch.setenv(name, "test-judge-key")
    else:
        (tmp_path / ".env").write_text(f"{name}=test-judge-key\n", encoding="utf-8")

    settings = load_settings(PROJECT_ROOT / "configs" / profile, project_root=tmp_path)
    assert settings.judge_llm is None


@pytest.mark.parametrize(
    ("field", "value", "missing"),
    [
        ("MODEL", "test-judge", "provider"),
        ("PROVIDER", "openai_compatible", "model"),
        ("BASE_URL", "https://judge.example.test/v1", "provider"),
    ],
)
def test_standalone_key_does_not_hide_an_incomplete_environment_judge(
    field: str, value: str, missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIAGAGENT_JUDGE_LLM__API_KEY", "test-judge-key")
    monkeypatch.setenv(f"DIAGAGENT_JUDGE_LLM__{field}", value)
    with pytest.raises(ConfigurationError, match=rf"judge_llm\.{missing}"):
        load_settings(project_root=PROJECT_ROOT, env_file=None)


@pytest.mark.parametrize("section", ["{}", "{api_key: yaml-test-key}"])
def test_explicit_incomplete_judge_yaml_is_rejected_with_standalone_key(
    section: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "judge.yaml"
    default = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    profile.write_text(f"{default}\njudge_llm: {section}\n", encoding="utf-8")
    monkeypatch.setenv("DIAGAGENT_JUDGE_LLM__API_KEY", "test-judge-key")
    with pytest.raises(ConfigurationError, match=r"judge_llm\.provider"):
        load_settings(profile, env_file=None)


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


def test_malformed_judge_yaml_does_not_expose_credentials_in_traceback(tmp_path: Path) -> None:
    secret = "test-malformed-yaml-key"
    profile = tmp_path / "malformed.yaml"
    profile.write_text(f"judge_llm: {{api_key: {secret}, model: [x}}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Could not read YAML profile") as failure:
        load_settings(profile, env_file=None)
    assert secret not in "".join(format_exception(failure.value))


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
