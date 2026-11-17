"""Common executor interface shared by all sandbox backends."""

from __future__ import annotations

import ast
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from types import TracebackType
from typing import Any

from coderl.data.schema import Problem
from coderl.sandbox import metrics
from coderl.sandbox.types import ExecutionResult, SandboxLimits
from coderl.utils.logging import get_logger

log = get_logger(__name__)


class SandboxUnavailableError(RuntimeError):
    """The requested backend cannot run on this machine (or not safely)."""


@lru_cache(maxsize=4096)
def protected_modules(test_setup: str, tests: tuple[str, ...]) -> tuple[str, ...]:
    """Stdlib modules the test code touches (``math.isclose(...)``, ``import math``).

    The harness imports these before the solution runs and restores their attributes before
    every test, so a solution cannot monkeypatch e.g. ``math.isclose`` to always return True.
    """
    names: set[str] = set()
    for src in (test_setup, *tests):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                names.add(node.value.id)
    return tuple(sorted(n for n in names if n in sys.stdlib_module_names and n != "builtins"))


def build_job(solution: str, problem: Problem) -> dict[str, Any]:
    tests = tuple(problem.tests)
    return {
        "solution": solution,
        "entry_point": problem.entry_point,
        "test_setup": problem.test_setup,
        "tests": list(tests),
        "protect_modules": list(protected_modules(problem.test_setup, tests)),
    }


class Executor(ABC):
    """Run untrusted solutions against a problem's unit tests."""

    name: str = "base"

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        self.limits = limits or SandboxLimits()

    # ------------------------------------------------------------------ public API
    def run(self, solution: str, problem: Problem) -> ExecutionResult:
        job = build_job(solution, problem)
        n = len(problem.tests)
        start = time.monotonic()
        metrics.INFLIGHT.labels(self.name).inc()
        try:
            raw = self._execute(job)
            result = ExecutionResult.model_validate({**raw, "backend": self.name})
        except Exception as exc:  # noqa: BLE001 - infrastructure failure is never a pass
            log.error("sandbox_error", backend=self.name, task_id=problem.task_id, error=repr(exc))
            result = ExecutionResult.sandbox_error(
                n,
                f"sandbox failure: {exc!r}",
                backend=self.name,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )
        finally:
            metrics.INFLIGHT.labels(self.name).dec()
        self._record(problem, result, time.monotonic() - start)
        return result

    def run_batch(
        self, items: Sequence[tuple[str, Problem]], max_workers: int | None = None
    ) -> list[ExecutionResult]:
        from coderl.sandbox.pool import run_batch

        return run_batch(self, items, max_workers or self.default_concurrency)

    @property
    def default_concurrency(self) -> int:
        from coderl.sandbox.pool import default_concurrency

        return default_concurrency()

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release backend resources (containers, temp dirs)."""

    def __enter__(self) -> Executor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ backend hook
    @abstractmethod
    def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        """Execute one job and return the runner's plain-dict result."""

    # ------------------------------------------------------------------ observability
    def _record(self, problem: Problem, result: ExecutionResult, seconds: float) -> None:
        metrics.EXECUTIONS.labels(self.name, result.status).inc()
        metrics.LATENCY.labels(self.name).observe(seconds)
        passed = result.num_passed
        metrics.TEST_CASES.labels(self.name, "passed").inc(passed)
        metrics.TEST_CASES.labels(self.name, "failed").inc(result.num_total - passed)
        for flag in result.flags:
            metrics.INTEGRITY_FLAGS.labels(self.name, flag).inc()
        if result.flags:
            log.warning(
                "sandbox_integrity_flags",
                backend=self.name,
                task_id=problem.task_id,
                flags=result.flags,
                status=result.status,
            )
        log.debug(
            "sandbox_execution",
            backend=self.name,
            task_id=problem.task_id,
            status=result.status,
            passed=passed,
            total=result.num_total,
            ms=round(seconds * 1000.0, 2),
        )
