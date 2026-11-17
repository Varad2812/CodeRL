"""Every canonical solution in data/processed must pass its own tests in the real sandbox.

A failure here means either a sandbox incompatibility or a Phase 1a test-splitting bug.
Skipped when data/processed has not been built (``make data``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coderl.data.schema import Problem, read_problems
from coderl.sandbox import Executor

from .conftest import REPO_ROOT

DATA = REPO_ROOT / "data" / "processed"
FILES = ("train.jsonl", "eval_mbpp_test.jsonl", "eval_humaneval.jsonl")
MIN_PASS_RATE = 1.0  # with the eval limit profile every reference solution must pass


def _problems() -> list[Problem]:
    paths = [DATA / f for f in FILES]
    if not all(p.exists() for p in paths):
        pytest.skip(f"processed data missing in {DATA} (run `make data`)")
    probs: list[Problem] = []
    for p in paths:
        probs.extend(read_problems(Path(p)))
    return probs


def test_canonical_solutions_pass(eval_executor: Executor) -> None:
    executor = eval_executor
    probs = [p for p in _problems() if p.canonical_solution]
    results = executor.run_batch([(p.canonical_solution or "", p) for p in probs])
    failures = [(p, r) for p, r in zip(probs, results, strict=True) if r.status != "passed"]
    lines = [
        f"{p.task_id}: {r.status} {r.num_passed}/{r.num_total} {r.error_type} "
        f"{(r.traceback.splitlines() or [''])[-1][:120]}"
        for p, r in failures
    ]
    rate = 1 - len(failures) / len(probs)
    print(f"\n[{executor.name}] canonical pass rate {rate:.2%} ({len(probs)} problems)")  # noqa: T201
    for line in lines:
        print("  FAIL", line)  # noqa: T201
    assert rate >= MIN_PASS_RATE, "\n".join(lines)
    flagged = [p.task_id for p, r in zip(probs, results, strict=True) if r.flags]
    assert not flagged, f"canonical solutions should raise no integrity flags: {flagged}"
