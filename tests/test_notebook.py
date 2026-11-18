"""The Kaggle notebook must be valid nbformat and every code cell must compile."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_eval_base.ipynb"


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
