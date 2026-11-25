from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

from src.core.exceptions import ConfigurationError


class ProfileSection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)


class DomainSettings(ProfileSection):
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)


class PathSettings(ProfileSection):
    corpus_dir: Path
    index_dir: Path
    reports_dir: Path


class IngestionSettings(ProfileSection):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "IngestionSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ModelSettings(ProfileSection):
    model: str = Field(min_length=1)


class LLMSettings(ModelSettings):
    provider: Literal["mock", "openai_compatible"]
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)


class EmbeddingSettings(ModelSettings):
    provider: Literal["mock", "openai_compatible", "sentence_transformers"]
    dimension: int = Field(gt=0)
    base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
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
    llm: LLMSettings
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

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
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read YAML profile {profile}") from exc

    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ConfigurationError(f"YAML profile {profile} must be a mapping with string keys")
    if any(key.startswith("_") for key in values):
        raise ConfigurationError(f"YAML profile {profile} contains a reserved setting name")

    try:
        settings = Settings(_env_file=root / env_file if env_file is not None else None, **values)
    except ValidationError as exc:
        fields = ", ".join(
            ".".join(str(part) for part in issue["loc"]) for issue in exc.errors()
        )
        raise ConfigurationError(f"Invalid settings in {profile}: {fields}") from None
    except (OSError, SettingsError) as exc:
        raise ConfigurationError(f"Could not load settings for {profile}") from exc

    settings.paths = PathSettings(
        corpus_dir=root / settings.paths.corpus_dir,
        index_dir=root / settings.paths.index_dir,
        reports_dir=root / settings.paths.reports_dir,
    )
    return settings
