"""Runtime settings for the telemedicine demo API."""

from __future__ import annotations

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - exercised only before dependencies are installed.
    BaseSettings = object
    SettingsConfigDict = dict


class Settings(BaseSettings):
    app_name: str = "MLOps Teledermatology Demo"
    database_url: str = "sqlite:///./storage/telemedicine_demo.db"
    jwt_secret_key: str = "change-me-for-demo"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 480
    upload_dir: Path = Path("storage/uploads")
    model_bundle_dir: Path = Path("storage/model_bundle")
    auto_build_model_bundle: bool = False
    model_mlflow_run_id: str = "ef084927bef741f996894b8a0fdd63e3"
    metadata_path: Path | None = None
    splits_dir: Path | None = None
    seed_demo_users: bool = True

    if hasattr(BaseSettings, "model_config"):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_prefix="TELEDERM_",
            protected_namespaces=(),
        )


def get_settings() -> Settings:
    return Settings()
