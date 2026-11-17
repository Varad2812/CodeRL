import json
from pathlib import Path

from typer.testing import CliRunner

from coderl.data.build import app
from coderl.data.schema import read_problems
from tests.conftest import HUMANEVAL_ROWS, MBPP_ROWS


def test_build_cli(fake_datasets: None, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    res = CliRunner().invoke(app, ["--out", str(out), "--threshold", "0.6"])
    assert res.exit_code == 0, res.output
    for name in [
        "train.jsonl",
        "eval_mbpp_test.jsonl",
        "eval_humaneval.jsonl",
        "contamination_report.json",
        "stats.json",
    ]:
        assert (out / name).exists(), name

    train = read_problems(out / "train.jsonl")
    ids = {p.task_id for p in train}
    assert "mbpp/11" not in ids  # collided with mbpp/801 -> removed from train
    assert {"mbpp/12", "mbpp/13", "mbpp/601"} <= ids
    assert len(read_problems(out / "eval_mbpp_test.jsonl")) == len(MBPP_ROWS["test"])  # untouched
    assert len(read_problems(out / "eval_humaneval.jsonl")) == len(HUMANEVAL_ROWS)

    report = json.loads((out / "contamination_report.json").read_text())
    assert "mbpp/11" in report["flagged_train_ids"]
    stats = json.loads((out / "stats.json").read_text())
    assert stats["counts"]["train_removed_contamination"] == 1
    assert stats["humaneval_split"] == {"split": 2, "fallback": 1, "success_rate": 0.6667}
    assert stats["avg_tests_per_problem"]["train"] > 0
