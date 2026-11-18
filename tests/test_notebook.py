"""The Kaggle notebook must be valid nbformat and every code cell must compile."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_eval_base.ipynb"


@pytest.mark.parametrize("limit", [10, None])
def test_run_configuration_routes_commands_and_archive(tmp_path: Path, limit: int | None) -> None:
    """A limited run must never use the full baseline's output paths."""
    nb = nbformat.read(NB, as_version=4)
    config = next(c.source for c in nb.cells if "REPO_URL =" in c.source)
    ns: dict = {}
    exec(config.replace("LIMIT = 10 ", f"LIMIT = {limit!r} "), ns)
    expected = "results/dry_run" if limit is not None else "results/base"
    assert ns["RESULTS_DIR"] == expected
    assert ns["GENERATION_BACKEND"] == "hf"

    # Define the evaluation helpers without starting any real GPU processes.
    source = next(c.source for c in nb.cells if "def eval_cmd(" in c.source)
    tree = ast.parse(source)
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns.update(PY="python", BACKEND="hf", N_GPUS=2)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "eval_helpers", "exec"), ns)
    cmd = ns["eval_cmd"]("qwen0.5b", "model")
    assert cmd[cmd.index("--out") + 1] == f"{expected}/qwen0.5b"
    assert cmd[cmd.index("--backend") + 1] == "hf"
    assert cmd[cmd.index("--hf-batch-size") + 1] == "20"
    assert cmd[cmd.index("--batch-size") + 1] == "8"
    assert "--resume" in cmd
    assert list(ns["MODELS"]) == ["qwen0.5b"], "run one model before downloading its results"
    assert ("--limit" in cmd) == (limit is not None)
    if limit is not None:
        assert cmd[cmd.index("--limit") + 1] == str(limit)

    # Execute the packaging cell with a temporary filesystem and fake archiver.
    import shutil
    from unittest.mock import patch

    ns["WORKDIR"] = str(tmp_path)
    import os

    ns["os"] = os
    archive = tmp_path / "result.zip"
    archive.write_bytes(b"zip")
    with patch.object(shutil, "make_archive", return_value=str(archive)) as make_archive:
        package = next(c.source for c in nb.cells if c.source.startswith("import shutil"))
        exec(package, ns)
    make_archive.assert_called_once_with(
        f"/kaggle/working/{ns['ARCHIVE_NAME']}", "zip", root_dir=str(tmp_path), base_dir=expected
    )


def test_notebook_packages_recovery_files_after_model_failure(tmp_path: Path) -> None:
    """A failed evaluation cell must still make the checkpoints easy to download."""
    import os
    import subprocess
    from types import SimpleNamespace
    from unittest.mock import patch

    nb = nbformat.read(NB, as_version=4)
    config = next(c.source for c in nb.cells if "REPO_URL =" in c.source)
    ns: dict = {"os": os, "subprocess": subprocess, "PY": "python", "BACKEND": "hf"}
    exec(config, ns)
    ns["WORKDIR"] = str(tmp_path)
    source = next(c.source for c in nb.cells if "def eval_cmd(" in c.source)
    failed = SimpleNamespace(returncode=1, poll=lambda: 1)
    gpu = SimpleNamespace(stdout="GPU 0: Tesla T4")
    with (
        patch("subprocess.run", return_value=gpu),
        patch("subprocess.Popen", return_value=failed),
        patch("shutil.make_archive", return_value="recovery.zip") as archive,
        patch("builtins.open", create=True) as opened,
        patch("os.makedirs"),
    ):
        opened.return_value.read.return_value = "simulated CUDA out of memory"
        with pytest.raises(RuntimeError, match="evaluation failed"):
            exec(source, ns)
    assert archive.call_count == 1
    assert archive.call_args.args[0].endswith("_recovery")


def test_hf_install_skips_vllm(tmp_path: Path) -> None:
    """Explicit HF mode must avoid the installation and startup that failed on T4."""
    from types import SimpleNamespace
    from unittest.mock import patch

    nb = nbformat.read(NB, as_version=4)
    install = next(c.source for c in nb.cells if "def sh(" in c.source)
    commands: list[str | list[str]] = []

    def run(cmd: str | list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    ns = {
        "WORKDIR": str(tmp_path),
        "REPO_REF": "main",
        "REPO_URL": "https://example.test/CodeRL",
        "GENERATION_BACKEND": "hf",
    }
    with patch("subprocess.run", side_effect=run), patch("os.chdir"):
        exec(install, ns)
    assert ns["BACKEND"] == "hf"
    assert any(".[hf]" in str(c) for c in commands)
    assert all(not isinstance(c, list) for c in commands), "must not probe/import vLLM"
    assert all(".[gen]" not in str(c) for c in commands)


def test_notebook_is_valid_and_cells_compile() -> None:
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert code, "notebook has no code cells"
    for i, src in enumerate(code):
        ast.parse(src, filename=f"cell {i}")  # no IPython magics: plain Python only
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"), "clear outputs"


def test_notebook_documents_kaggle_settings_and_runs_both_models() -> None:
    nb = nbformat.read(NB, as_version=4)
    text = "\n".join(c.source for c in nb.cells)
    for needle in (
        "GPU T4 x2",
        "Internet",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "coderl.eval.run_benchmarks",
        "coderl.eval.report",
        "coderl_results_base",
        "bubblewrap",
        '"float16"',
    ):
        assert needle in text, needle
