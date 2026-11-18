"""End-to-end runner tests with MockGenerator. Fully offline.

``FakeExecutor`` runs (trusted, hand-written) code in-process so the pipeline is tested on every
platform; ``test_end_to_end_real_sandbox`` repeats it through the real sandbox backends.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from coderl.data.schema import Problem, write_problems
from coderl.eval.report import load_summaries, render_table
from coderl.eval.run_benchmarks import EvalConfig, run_eval
from coderl.gen.mock import MockGenerator, canonical_responder, fenced
from coderl.sandbox import Executor

from ..sandbox.conftest import BACKEND_PARAMS

HE = [
    Problem(
        task_id="humaneval/0",
        source="humaneval",
        split="test",
        prompt="from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    '''Sum.'''\n",
        entry_point="add",
        tests=["assert add([1, 2]) == 3", "assert add([]) == 0"],
        canonical_solution="from typing import List\n\n\ndef add(xs: List[int]) -> int:\n"
        "    return sum(xs)\n",
    ),
    Problem(
        task_id="humaneval/1",
        source="humaneval",
        split="test",
        prompt="def neg(x):\n    '''Negate.'''\n",
        entry_point="neg",
        tests=["assert neg(1) == -1"],
        canonical_solution="def neg(x):\n    return -x\n",
    ),
]
MBPP = [
    Problem(
        task_id="mbpp/1",
        source="mbpp",
        split="test",
        prompt="Write a function to double.\nYour code should pass this test:\n"
        "assert double(2) == 4",
        entry_point="double",
        tests=["assert double(2) == 4", "assert double(0) == 0"],
        canonical_solution="def double(x):\n    return 2 * x\n",
    ),
    Problem(
        task_id="mbpp/2",
        source="mbpp",
        split="test",
        prompt="Write a function to loop forever.\nYour code should pass this test:\n"
        "assert spin() == 1",
        entry_point="spin",
        tests=["assert spin() == 1"],
        canonical_solution="def spin():\n    return 1\n",
    ),
]


class FakeExecutor(Executor):
    """In-process exec of trusted test code (NOT a sandbox)."""

    name = "fake"

    def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        ns: dict[str, Any] = {}
        try:
            compile(job["solution"], "<solution>", "exec")
        except SyntaxError:
            return {"status": "syntax_error", "num_total": len(job["tests"])}
        exec(job["test_setup"] + "\n" + job["solution"], ns)
        tests = []
        for i, t in enumerate(job["tests"]):
            try:
                exec(t, ns)
                tests.append({"index": i, "passed": True})
            except Exception as exc:  # noqa: BLE001
                tests.append({"index": i, "passed": False, "error_type": type(exc).__name__})
        n_ok = sum(t["passed"] for t in tests)
        return {
            "status": "passed" if n_ok == len(tests) else "failed",
            "tests": tests,
            "num_passed": n_ok,
            "num_total": len(tests),
        }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    write_problems(d / "eval_humaneval.jsonl", HE)
    write_problems(d / "eval_mbpp_test.jsonl", MBPP)
    return d


def _cfg(tmp_path: Path, data_dir: Path, **kw: Any) -> EvalConfig:
    kw.setdefault("batch_size", 1)
    return EvalConfig(model="mock/model", out=tmp_path / "out", data_dir=data_dir, **kw)


def test_end_to_end_all_correct(tmp_path: Path, data_dir: Path) -> None:
    gen = MockGenerator(canonical_responder(HE + MBPP))
    summaries = run_eval(gen, _cfg(tmp_path, data_dir, n_samples=4), FakeExecutor())
    assert [s["benchmark"] for s in summaries] == ["humaneval", "mbpp_test"]
    for s in summaries:
        assert s["metrics"] == {
            "pass@1_greedy": 1.0,
            "pass@1_sampled": 1.0,
            "pass@10_sampled": None,
        }
        assert s["passes"]["sampled"]["diagnostics"]["num_completions"] == 4 * 2
        assert s["passes"]["sampled"]["unique_programs_executed"] == 2  # identical code cached
    # greedy pass really is greedy, seeds are per-problem offsets
    assert gen.calls[0] == {"n_prompts": 1, "n": 1, "temperature": 0.0, "seed": 1234}
    assert gen.calls[1]["seed"] == 1235


def test_pass_at_k_with_partial_correctness(tmp_path: Path, data_dir: Path) -> None:
    # every 4th sample (j = 0, 4, 8, ...) is wrong -> with n=20: c = 15 per problem
    gen = MockGenerator(canonical_responder(HE + MBPP, wrong_every=4))
    cfg = _cfg(tmp_path, data_dir, benchmarks=["mbpp_test"], n_samples=20)
    (s,) = run_eval(gen, cfg, FakeExecutor())
    assert s["metrics"]["pass@1_greedy"] == 0.0  # greedy = sample 0 = wrong
    assert s["metrics"]["pass@1_sampled"] == pytest.approx(15 / 20)
    assert s["metrics"]["pass@10_sampled"] == 1.0  # only 5 wrong < k

    d = s["passes"]["sampled"]["diagnostics"]
    assert d["status_counts"]["passed"] == 30 and d["status_counts"]["failed"] == 10
    assert d["extraction_status"] == {"code_block": 40}


def test_outputs_files_and_report(tmp_path: Path, data_dir: Path) -> None:
    gen = MockGenerator(lambda p, j: "no code here, sorry" if j else fenced("def f(:\n"))
    cfg = _cfg(tmp_path, data_dir, n_samples=2, limit=1)
    run_eval(gen, cfg, FakeExecutor())
    he_dir = tmp_path / "out" / "humaneval"
    summary = json.loads((he_dir / "summary.json").read_text())
    assert summary["num_problems"] == 1 and summary["limit"] == 1
    assert summary["sandbox"]["backend"] == "fake" and summary["sandbox"]["profile"] == "eval"
    assert summary["env"]["python"] and "packages" in summary["env"]
    assert summary["passes"]["greedy"]["diagnostics"]["syntax_error_rate"] == 1.0
    with gzip.open(he_dir / "samples.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 1 + 2
    assert {r["extraction_status"] for r in rows} == {"code_block", "raw"}
    assert all("result" in r and "code" in r and "completion" in r for r in rows)

    table = render_table(load_summaries(tmp_path / "out"))
    assert "| model | — | humaneval | 1 (limit) |" in table
    assert table.count("\n| model ") == 2


def test_flagged_results_never_count(tmp_path: Path, data_dir: Path) -> None:
    class Flagging(FakeExecutor):
        def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
            return {**super()._execute(job), "flags": ["custom_eq"]}

    gen = MockGenerator(canonical_responder(HE + MBPP))
    cfg = _cfg(tmp_path, data_dir, benchmarks=["humaneval"], n_samples=1)
    (s,) = run_eval(gen, cfg, Flagging())
    assert s["metrics"]["pass@1_greedy"] == 0.0
    assert s["passes"]["greedy"]["diagnostics"]["flagged_results"] == 2


@pytest.fixture(params=BACKEND_PARAMS)
def real_executor(request: pytest.FixtureRequest) -> Iterator[Executor]:
    from coderl.sandbox.types import load_limits

    if request.param == "docker":
        from coderl.sandbox.backends.docker_backend import DockerBackend

        ex: Executor = DockerBackend(load_limits("eval"), pool_size=2)
    else:
        from coderl.sandbox.backends.subprocess_backend import SubprocessBackend

        ex = SubprocessBackend(load_limits("eval"))
    yield ex
    ex.close()


def test_end_to_end_real_sandbox(tmp_path: Path, data_dir: Path, real_executor: Executor) -> None:
    def respond(prompt: str, j: int) -> str:
        if "spin" in prompt:  # an infinite loop must be a timeout, not a hang or a pass
            return fenced("def spin():\n    while True:\n        pass\n")
        return canonical_responder(HE + MBPP)(prompt, j)

    gen = MockGenerator(respond)
    cfg = _cfg(tmp_path, data_dir, n_samples=2, batch_size=2)
    he, mb = run_eval(gen, cfg, real_executor)
    assert he["metrics"]["pass@1_greedy"] == 1.0 and he["metrics"]["pass@1_sampled"] == 1.0
    assert mb["metrics"]["pass@1_greedy"] == 0.5
    assert mb["passes"]["greedy"]["diagnostics"]["status_counts"]["timeout"] == 1
    assert he["sandbox"]["isolation"] in ("bwrap", "unshare", "none", "docker")
