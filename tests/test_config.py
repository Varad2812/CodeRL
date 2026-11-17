from pathlib import Path

import pytest

from coderl.utils.config import load_settings


def test_defaults_yaml_and_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CODERL_LOG_LEVEL", "CODERL_DATA_DIR", "WANDB_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert load_settings().log_level == "INFO"

    cfg = tmp_path / "c.yaml"
    cfg.write_text("log_level: DEBUG\ndata_dir: /tmp/x\n")
    s = load_settings(cfg)
    assert s.log_level == "DEBUG" and s.data_dir == Path("/tmp/x")

    monkeypatch.setenv("CODERL_LOG_LEVEL", "WARNING")
    assert load_settings(cfg).log_level == "WARNING"  # env overrides YAML


def test_yaml_must_be_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_settings(cfg)
