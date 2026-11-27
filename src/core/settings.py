from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

from src.core.exceptions import ConfigurationError

FAULT_TABLE = "fault_codes.csv"
SENSOR_TABLE = "sensor_thresholds.json"


class ProfileSection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)


class DomainSettings(ProfileSection):
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)


class PathSettings(ProfileSection):
    corpus_dir: Path
    index_dir: Path
    reports_dir: Path
    cache_file: Path = Path("data/cache/semantic.sqlite3")

    @field_validator("corpus_dir", "index_dir", "reports_dir", "cache_file", mode="before")
    @classmethod
    def reject_blank_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Path must not be blank")
        return value


class IngestionSettings(ProfileSection):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    prune_generations_on_ingest: bool = False

    @model_validator(mode="after")
    def validate_overlap(self) -> "IngestionSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RulesSettings(ProfileSection):
    fault_codes_path: str = FAULT_TABLE
    sensor_thresholds_path: str = SENSOR_TABLE
    allow_missing_rule_tables: bool = False

    @model_validator(mode="after")
    def validate_source_paths(self) -> "RulesSettings":
        for value, suffix in (
            (self.fault_codes_path, ".csv"),
            (self.sensor_thresholds_path, ".json"),
        ):
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or path.as_posix() != value
                or any(part.startswith(".") for part in path.parts)
                or any(char in value for char in ("\\", ":", "\x00"))
                or path.suffix.lower() != suffix
            ):
                raise ValueError(f"Rule table paths must be corpus-relative {suffix} paths")
        return self

    @property
    def source_paths(self) -> tuple[str, str]:
        return self.fault_codes_path, self.sensor_thresholds_path


class RetrievalSettings(ProfileSection):
    candidate_k: int = Field(default=20, gt=0)
    rrf_rank_constant: int = Field(default=60, gt=0)


class CacheSettings(ProfileSection):
    similarity_threshold: float = Field(default=0.95, ge=0.95, le=1)
    ttl_seconds: int | None = Field(default=None, gt=0)
    max_entries: int | None = Field(default=10000, gt=0)


class ApiSettings(ProfileSection):
    api_key: SecretStr | None = None
    admin_api_key: SecretStr | None = None
    cors_allow_origins: tuple[str, ...] = ()
    max_query_chars: int = Field(default=4000, gt=0)
    max_sensor_readings: int = Field(default=64, ge=0)
    rate_limit_rpm: int | None = Field(default=None, gt=0)
    enable_metrics: bool = False
    enable_demo_ui: bool = True

    @field_validator("api_key", "admin_api_key", mode="before")
    @classmethod
    def unset_empty_key(cls, value: object) -> object:
        if isinstance(value, str) and value == "":
            return None
        return value

    @field_validator("api_key", "admin_api_key")
    @classmethod
    def require_nonblank_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("API keys must not be blank")
        return value


class ModelSettings(ProfileSection):
    model: str = Field(min_length=1)


class LLMSettings(ModelSettings):
    provider: Literal["mock", "openai_compatible"]
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    json_mode: bool = False

    @model_validator(mode="after")
    def validate_token_limit(self) -> "LLMSettings":
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Set only one of max_tokens or max_completion_tokens")
        return self


class EmbeddingSettings(ModelSettings):
    provider: Literal["mock", "openai_compatible", "sentence_transformers"]
    dimension: int = Field(gt=0)
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    batch_size: int = Field(default=32, gt=0)
    request_dimensions: bool = False
    device: str | None = None


class RerankerSettings(ModelSettings):
    provider: Literal["mock", "sentence_transformers"]
    device: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DIAGAGENT_",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    domain: DomainSettings
    paths: PathSettings
    ingestion: IngestionSettings
    rules: RulesSettings = Field(default_factory=RulesSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    llm: LLMSettings
    judge_llm: LLMSettings | None = None
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("judge_llm", mode="before")
    @classmethod
    def ignore_standalone_judge_key(cls, value: object) -> object:
        # A key can be supplied in advance for replay with an explicit model and gateway.
        if isinstance(value, Mapping) and set(value) == {"api_key"}:
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return env_settings, dotenv_settings, init_settings, file_secret_settings


def load_settings(
    config_path: str | Path = "configs/default.yaml",
    *,
    project_root: str | Path | None = None,
    env_file: str | Path | None = ".env",
) -> Settings:
    """Load a YAML profile with environment and dotenv overrides.

    Relative profile, dotenv, and application paths are rooted at the project directory.
    """
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    profile = root / config_path
    try:
        with profile.open(encoding="utf-8-sig") as stream:
            values = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigurationError(f"Could not read YAML profile {profile}") from None

    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ConfigurationError(f"YAML profile {profile} must be a mapping with string keys")
    if any(key.startswith("_") for key in values):
        raise ConfigurationError(f"YAML profile {profile} contains a reserved setting name")

    try:
        settings = Settings(_env_file=root / env_file if env_file is not None else None, **values)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in issue["loc"]) for issue in exc.errors())
        raise ConfigurationError(f"Invalid settings in {profile}: {fields}") from None
    except (OSError, UnicodeError, SettingsError):
        raise ConfigurationError(f"Could not load settings for {profile}") from None

    if settings.judge_llm is None and isinstance(values.get("judge_llm"), dict):
        # An explicit YAML section must be complete after environment overrides are merged.
        raise ConfigurationError(
            f"Invalid settings in {profile}: judge_llm.provider, judge_llm.model"
        )

    settings.paths = PathSettings(
        corpus_dir=root / settings.paths.corpus_dir,
        index_dir=root / settings.paths.index_dir,
        reports_dir=root / settings.paths.reports_dir,
        cache_file=root / settings.paths.cache_file,
    )
    return settings
