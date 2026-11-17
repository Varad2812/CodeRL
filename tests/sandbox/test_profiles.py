from pathlib import Path

import pytest

from coderl.sandbox.types import SandboxLimits, load_limits


def test_builtin_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODERL_SANDBOX_CONFIG", raising=False)
    assert load_limits("train") == SandboxLimits()
    ev = load_limits("eval")
    assert (ev.cpu_seconds, ev.wall_seconds, ev.per_test_seconds) == (6, 12.0, 4.0)
    assert ev.memory_mb == SandboxLimits().memory_mb
    with pytest.raises(ValueError):
        load_limits("nope")


def test_yaml_then_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "sandbox.yaml"
    cfg.write_text("profiles:\n  eval:\n    cpu_seconds: 9\n    wall_seconds: 20\n")
    monkeypatch.setenv("CODERL_SANDBOX_CONFIG", str(cfg))
    assert load_limits("eval").cpu_seconds == 9
    monkeypatch.setenv("CODERL_SANDBOX_EVAL_CPU_SECONDS", "7")
    ev = load_limits("eval")
    assert ev.cpu_seconds == 7 and ev.wall_seconds == 20.0
    assert load_limits("train").cpu_seconds == 2
