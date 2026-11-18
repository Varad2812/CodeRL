"""Benchmark the sandbox backends on all MBPP-test canonical solutions.

    uv run python scripts/bench_sandbox.py [--workers 1,4,8] [--out results/sandbox_bench.json]

For every available backend and concurrency level: problems/sec, test cases/sec and host-side
end-to-end latency percentiles (p50/p95), measured around ``Executor.run`` as called by
``run_batch``.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from coderl.data.schema import Problem, read_problems
from coderl.sandbox import ExecutionResult, Executor, SandboxLimits
from coderl.utils.logging import configure_logging

app = typer.Typer(add_completion=False)
DATA = Path("data/processed/eval_mbpp_test.jsonl")


def _percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _make(backend: str, workers: int) -> Executor:
    if backend == "docker":
        from coderl.sandbox.backends.docker_backend import DockerBackend

        return DockerBackend(SandboxLimits(), pool_size=workers)
    from coderl.sandbox.backends.subprocess_backend import SubprocessBackend

    return SubprocessBackend(SandboxLimits())


def _available() -> list[str]:
    out = []
    if sys.platform == "linux":
        out.append("subprocess")
    from coderl.sandbox.backends.docker_backend import docker_available

    if docker_available():
        out.append("docker")
    return out


def bench_one(backend: str, workers: int, problems: list[Problem]) -> dict[str, Any]:
    ex = _make(backend, workers)
    latencies: list[float] = []
    lock = threading.Lock()
    run = ex.run

    def timed_run(solution: str, problem: Problem) -> ExecutionResult:
        t0 = time.perf_counter()
        try:
            return run(solution, problem)
        finally:
            with lock:
                latencies.append((time.perf_counter() - t0) * 1000.0)

    ex.run = timed_run  # type: ignore[method-assign]
    items = [(p.canonical_solution or "", p) for p in problems]
    try:
        ex.run_batch(items[: min(40, len(items))], max_workers=workers)  # warm-up
        latencies.clear()
        t0 = time.perf_counter()
        results = ex.run_batch(items, max_workers=workers)
        elapsed = time.perf_counter() - t0
    finally:
        ex.close()
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    n_tests = sum(r.num_total for r in results)
    return {
        "backend": backend,
        "workers": workers,
        "problems": len(results),
        "test_cases": n_tests,
        "seconds": round(elapsed, 3),
        "problems_per_sec": round(len(results) / elapsed, 1),
        "test_cases_per_sec": round(n_tests / elapsed, 1),
        "latency_ms_p50": round(_percentile(latencies, 0.50), 1),
        "latency_ms_p95": round(_percentile(latencies, 0.95), 1),
        "latency_ms_mean": round(statistics.fmean(latencies), 1),
        "statuses": statuses,
    }


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            pass
    try:
        info["docker"] = subprocess.run(
            [
                "docker",
                "version",
                "--format",
                "{{.Server.Version}} ({{.Server.Os}}/{{.Server.Arch}})",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


@app.command()
def main(
    workers: Annotated[str, typer.Option(help="Comma-separated concurrency levels.")] = "1,4,8",
    out: Annotated[Path, typer.Option(help="JSON output path.")] = Path(
        "results/sandbox_bench.json"
    ),
    note: Annotated[str, typer.Option(help="Free-text note stored with the results.")] = "",
) -> None:
    configure_logging("WARNING")
    if not DATA.exists():
        raise typer.BadParameter(f"{DATA} missing; run `make data` first")
    problems = [p for p in read_problems(DATA) if p.canonical_solution]
    levels = [int(w) for w in workers.split(",") if w.strip()]
    rows = []
    for backend in _available():
        for w in levels:
            row = bench_one(backend, w, problems)
            rows.append(row)
            sys.stdout.write(f"  done: {backend} x{w}: {row['problems_per_sec']} problems/s\n")
    header = (
        f"{'backend':<11}{'workers':>8}{'problems/s':>12}{'tests/s':>10}"
        f"{'p50 ms':>9}{'p95 ms':>9}  statuses"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['backend']:<11}{r['workers']:>8}{r['problems_per_sec']:>12}"
            f"{r['test_cases_per_sec']:>10}{r['latency_ms_p50']:>9}{r['latency_ms_p95']:>9}"
            f"  {r['statuses']}"
        )
    sys.stdout.write("\n".join(lines) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(DATA),
        "limits": SandboxLimits().model_dump(),
        "machine": _machine(),
        "note": note,
        "results": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stdout.write(f"wrote {out}\n")


if __name__ == "__main__":
    app()
