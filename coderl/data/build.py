"""Build the processed datasets: ``python -m coderl.data.build --out data/processed``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from coderl.data.dedup import DEFAULT_THRESHOLD, check_contamination, remove_contaminated
from coderl.data.loaders import load_source
from coderl.data.schema import Problem, write_problems
from coderl.utils.logging import get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Build CodeRL train/eval datasets.")


def _avg_tests(problems: list[Problem]) -> float:
    return round(sum(len(p.tests) for p in problems) / len(problems), 3) if problems else 0.0


def build(out: Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, object]:
    """Load, dedup and write all datasets. Returns the stats dict."""
    out.mkdir(parents=True, exist_ok=True)

    # TRAIN = MBPP train + validation; EVAL = MBPP test + HumanEval (never modified)
    train_raw = load_source("mbpp", "train") + load_source("mbpp", "validation")
    eval_mbpp = load_source("mbpp", "test")
    eval_he = load_source("humaneval", "test")

    report = check_contamination(train_raw, eval_mbpp + eval_he, threshold=threshold)
    train = remove_contaminated(train_raw, report)

    write_problems(out / "train.jsonl", train)
    write_problems(out / "eval_mbpp_test.jsonl", eval_mbpp)
    write_problems(out / "eval_humaneval.jsonl", eval_he)
    (out / "contamination_report.json").write_text(report.model_dump_json(indent=2))

    he_split = sum(p.metadata.get("test_split") == "split" for p in eval_he)
    stats: dict[str, object] = {
        "counts": {
            "train_before_dedup": len(train_raw),
            "train_after_dedup": len(train),
            "train_removed_contamination": len(train_raw) - len(train),
            "eval_mbpp_test": len(eval_mbpp),
            "eval_humaneval": len(eval_he),
        },
        "avg_tests_per_problem": {
            "train": _avg_tests(train),
            "eval_mbpp_test": _avg_tests(eval_mbpp),
            "eval_humaneval": _avg_tests(eval_he),
        },
        "humaneval_split": {
            "split": he_split,
            "fallback": len(eval_he) - he_split,
            "success_rate": round(he_split / len(eval_he), 4) if eval_he else 0.0,
        },
        "dedup_threshold": threshold,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    log.info("build_complete", out=str(out), stats=stats)
    return stats


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/processed"),
    threshold: Annotated[
        float, typer.Option(help="Near-duplicate Jaccard threshold.")
    ] = DEFAULT_THRESHOLD,
) -> None:
    build(out, threshold)


if __name__ == "__main__":
    app()
