"""Central configuration: filesystem paths, environment variables and YAML configs.

Import ``get_settings()`` anywhere in the package to get a cached ``Settings``
instance. Environment variables (optionally from a ``.env`` file at the project
root) override the defaults below; the ``configs/*.yaml`` files hold the richer,
non-secret configuration and are loaded on demand via ``load_config()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so they work regardless of CWD)
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent           # src/energy_forecasting
PROJECT_ROOT = PACKAGE_ROOT.parents[1]                    # repository root
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class Settings(BaseSettings):
    """Environment-driven settings. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # NESO Historic Demand Data (CKAN)
    neso_base_url: str = "https://api.neso.energy"
    neso_demand_resource_id: str | None = None  # optional single-resource override

    # Weather
    weather_provider: str = "open-meteo"
    weather_api_key: str | None = None

    # Storage
    database_url: str = "sqlite:///data/processed/energy.db"

    # App / API / logging
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    log_level: str = "INFO"

    # Convenience path attributes (not env-driven)
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)
    config_dir: Path = Field(default=CONFIG_DIR, exclude=True)
    raw_dir: Path = Field(default=RAW_DIR, exclude=True)
    interim_dir: Path = Field(default=INTERIM_DIR, exclude=True)
    processed_dir: Path = Field(default=PROCESSED_DIR, exclude=True)
    external_dir: Path = Field(default=EXTERNAL_DIR, exclude=True)
    models_dir: Path = Field(default=MODELS_DIR, exclude=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()


def load_config(name: str) -> dict[str, Any]:
    """Load and parse a YAML file from ``configs/``.

    ``name`` may be given with or without the ``.yaml`` suffix, e.g.
    ``load_config("data")`` or ``load_config("data.yaml")``.
    """
    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
