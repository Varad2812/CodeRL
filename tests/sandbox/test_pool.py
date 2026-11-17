"""Fast unit tests: worker pool, backend selection, job building and the runner protocol.

These need no Docker and no namespaces, so they run everywhere (incl. macOS and CI job 1).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from typing import Any

import pytest

from coderl.sandbox import ExecutionResult, Executor, SandboxLimits, SandboxUnavailableError, runner
from coderl.sandbox.backends import default_backend_name, get_executor
from coderl.sandbox.backends.base import build_job, protected_modules
from coderl.sandbox.backends.subprocess_backend import HARNESS_PATH, SubprocessBackend, base_python
from coderl.sandbox.metrics import EXECUTIONS

from .conftest import problem


class FakeExecutor(Executor):
    """Passes iff the solution string is 'ok'; 'crash' raises; 'slow' sleeps."""

    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if job["solution"] == "crash":
                raise RuntimeError("worker exploded")
            time.sleep(0.02)
            ok = job["solution"] == "ok"
            n = len(job["tests"])
            return {
                "status": "passed" if ok else "failed",
                "tests": [{"index": i, "passed": ok} for i in range(n)],
                "num_passed": n if ok else 0,
                "num_total": n,
            }
        finally:
            with self._lock:
                self.active -= 1


def test_run_batch_preserves_order_and_bounds_concurrency() -> None:
    ex = FakeExecutor()
    items = [("ok" if i % 3 else "bad", problem([f"assert f({i})"])) for i in range(30)]
    results = ex.run_batch(items, max_workers=4)
    assert [r.status for r in results] == ["passed" if i % 3 else "failed" for i in range(30)]
    assert ex.max_active <= 4


def test_crash_is_isolated_to_one_item() -> None:
    ex = FakeExecutor()
    items = [("ok", problem(["a", "b"])), ("crash", problem(["a", "b"])), ("ok", problem(["a"]))]
    results = ex.run_batch(items, max_workers=3)
    assert [r.status for r in results] == ["passed", "sandbox_error", "passed"]
    crashed = results[1]
    assert crashed.num_passed == 0 and crashed.num_total == 2
    assert all(not t.passed for t in crashed.tests)


def test_metrics_count_statuses() -> None:
    before = EXECUTIONS.labels("fake", "passed")._value.get()
    FakeExecutor().run("ok", problem(["a"]))
    assert EXECUTIONS.labels("fake", "passed")._value.get() == before + 1


def test_sandbox_error_never_passes() -> None:
    r = ExecutionResult.sandbox_error(3, "boom")
    assert r.status == "sandbox_error" and r.num_passed == 0 and r.pass_fraction == 0.0


def test_protected_modules_found_from_tests() -> None:
    tests = ("assert math.isclose(f(1), 2.0)", "assert sys.getsizeof(f(1)) > 0")
    assert protected_modules("import random\nfrom collections import Counter", tests) == (
        "collections",
        "math",
        "random",
        "sys",
    )
    assert protected_modules("", ("assert x.append(1) is None",)) == ()


def test_build_job_carries_everything_but_the_canonical_solution() -> None:
    job = build_job("def f(): pass", problem(["assert f() is None"], test_setup="import math"))
    assert job["solution"] == "def f(): pass"
    assert job["protect_modules"] == ["math"]
    assert set(job) == {"solution", "entry_point", "test_setup", "tests", "protect_modules"}


# ---------------------------------------------------------------------- backend selection
def test_default_backend_per_platform() -> None:
    assert default_backend_name() == ("subprocess" if sys.platform == "linux" else "docker")


def test_unknown_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODERL_SANDBOX_BACKEND", "chroot")
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        get_executor()


def test_subprocess_backend_refuses_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("CODERL_ALLOW_UNSAFE_SUBPROCESS", raising=False)
    with pytest.raises(SandboxUnavailableError, match="Linux-only"):
        SubprocessBackend()


# ---------------------------------------------------------------------- runner protocol
def _run_direct(solution: str, tests: list[str], **limits: Any) -> dict[str, Any]:
    """Harness + runner with no OS isolation at all (trusted code only)."""
    lim = SandboxLimits(**limits).wire()
    python, _ = base_python()
    with tempfile.TemporaryDirectory() as d:
        hp = runner.spawn_harness([python, "-I", "-S", "-u", str(HARNESS_PATH)], lim, cwd=d)
        job = {"solution": solution, "entry_point": "f", "tests": tests, "test_setup": ""}
        return runner.execute(hp, job, lim)


def test_runner_happy_path() -> None:
    r = _run_direct("def f(x):\n    return x\n", ["assert f(1) == 1", "assert f(2) == 3"])
    assert r["status"] == "failed" and r["num_passed"] == 1
    assert r["tests"][1]["error_type"] == "AssertionError"


def test_runner_rejects_records_without_nonce() -> None:
    sol = "def f(x):\n    return 0\n"
    forge = "".join(
        f"try:\n    os.write({fd}, b'deadbeef\\tD\\t\\n')\nexcept OSError:\n    pass\n"
        for fd in range(3, 64)
    )
    r = _run_direct(sol + "import os\n" + forge, ["assert f(1) == 1"])
    assert r["num_passed"] == 0
    assert "forged_records" in r["flags"]


def test_runner_early_exit_fails_all() -> None:
    r = _run_direct("import os\ndef f(x):\n    os._exit(0)\n", ["assert f(1) == 1"] * 3)
    assert r["status"] == "runtime_error" and r["error_type"] == "EarlyExit"
    assert r["num_passed"] == 0


def test_runner_wall_timeout() -> None:
    start = time.monotonic()
    r = _run_direct("import time\ntime.sleep(30)\n", ["assert f(1) == 1"], wall_seconds=1.0)
    assert r["status"] == "timeout"
    assert time.monotonic() - start < 3


def test_harness_marker_present() -> None:
    assert HARNESS_PATH.read_text().startswith("# coderl-sandbox-harness")
    assert os.access(HARNESS_PATH, os.R_OK)
