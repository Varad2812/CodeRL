from pathlib import Path

import pytest

from coderl.data.schema import Problem, read_problems, write_problems


def _p(i: int) -> Problem:
    return Problem(
        task_id=f"mbpp/{i}",
        source="mbpp",
        split="train",
        prompt="do it\nunicode: é",
        entry_point="f",
        tests=["assert f(1) == 1", "assert f(2) == 2"],
        test_setup="import math",
        canonical_solution="def f(x):\n    return x",
        metadata={"k": [1, 2], "flag": True},
    )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    probs = [_p(1), _p(2)]
    path = tmp_path / "sub" / "p.jsonl"
    assert write_problems(path, probs) == 2
    assert read_problems(path) == probs


def test_invalid_source_rejected() -> None:
    with pytest.raises(ValueError):
        Problem(task_id="x/1", source="nope", split="train", prompt="", entry_point="f", tests=[])  # type: ignore[arg-type]
