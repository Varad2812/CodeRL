"""Settings loader: defaults < optional YAML file < environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    wandb_api_key: str | None = None
    hf_token: str | None = None


_ENV_MAP = {
    "log_level": "CODERL_LOG_LEVEL",
    "data_dir": "CODERL_DATA_DIR",
    "wandb_api_key": "WANDB_API_KEY",
    "hf_token": "HF_TOKEN",
}


def load_settings(yaml_path: str | Path | None = None) -> Settings:
    """Load settings; environment variables override YAML values."""
    values: dict[str, Any] = {}
    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            values.update(loaded)
    for field, env in _ENV_MAP.items():
        val = os.environ.get(env)
        if val:
            values[field] = val
    return Settings(**values)
