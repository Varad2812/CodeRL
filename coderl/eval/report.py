"""Aggregate ``summary.json`` files into a markdown table.

python -m coderl.eval.report results/            # every summary.json under results/
python -m coderl.eval.report results/ --out results/README.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(add_completion=False, help="Markdown table of evaluation results.")

COLUMNS = (
    "Model",
    "Adapter",
    "Benchmark",
    "Problems",
    "pass@1 greedy",
    "pass@1 (T=0.8)",
    "pass@10 (T=0.8)",
    "n",
    "Syntax err",
    "Timeout",
    "Sandbox",
)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def load_summaries(root: Path) -> list[dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(root.rglob("summary.json"))
    out = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "metrics" in data and "benchmark" in data:
            out.append(data)
    return out


def _row(s: dict[str, Any]) -> list[str]:
    m = s["metrics"]
    passes = s.get("passes", {})
    diag = (passes.get("sampled") or passes.get("greedy") or {}).get("diagnostics", {})
    sb = s.get("sandbox", {})
    model = s["model"].split("/")[-1]
    return [
        model,
        Path(s["adapter"]).name if s.get("adapter") else "—",
        s["benchmark"],
        str(s["num_problems"]) + (" (limit)" if s.get("limit") else ""),
        _pct(m.get("pass@1_greedy")),
        _pct(m.get("pass@1_sampled")),
        _pct(m.get("pass@10_sampled")),
        str(passes.get("sampled", {}).get("n", "—")),
        _pct(diag.get("syntax_error_rate")),
        _pct(diag.get("timeout_rate")),
        f"{sb.get('backend', '?')}/{sb.get('isolation', '?')}",
    ]


def render_table(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "_no results_"
    rows = sorted(
        (_row(s) for s in summaries), key=lambda r: (r[2], r[0], r[1])
    )  # by benchmark, model, adapter
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" if i < 3 else "---:" for i in range(len(COLUMNS))) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines.append("")
    lines.append("pass@k and error rates in %. Error rates are over sampled completions.")
    return "\n".join(lines)


@app.command()
def main(
    root: Annotated[Path, typer.Argument(help="Results dir (searched recursively) or file.")],
    out: Annotated[Path | None, typer.Option(help="Also write the table here.")] = None,
) -> None:
    table = render_table(load_summaries(root))
    sys.stdout.write(table + "\n")
    if out:
        out.write_text(table + "\n")


if __name__ == "__main__":
    app()
