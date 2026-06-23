from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application environment/settings using pydantic v2 style.

    Adjust or add fields here for any environment-driven config values.
    """

    DEBUG: bool = False
    DATA_PATH: Path = None

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[2] / ".env",
        env_file_encoding="utf-8",
        env_prefix="DAGGER_",
    )


settings = AppConfig()
