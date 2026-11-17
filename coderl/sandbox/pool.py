"""Bounded-concurrency batch execution with per-item failure isolation."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from coderl.data.schema import Problem
from coderl.sandbox.types import ExecutionResult
from coderl.utils.logging import get_logger

if TYPE_CHECKING:
    from coderl.sandbox.backends.base import Executor

log = get_logger(__name__)


def default_concurrency() -> int:
    return max(1, os.cpu_count() or 1)


def run_batch(
    executor: Executor,
    items: Sequence[tuple[str, Problem]],
    max_workers: int | None = None,
) -> list[ExecutionResult]:
    """Run ``(solution, problem)`` pairs concurrently; results are returned in input order.

    A crash while handling one item becomes a ``sandbox_error`` result for that item only;
    it never aborts the batch. Workers are threads: the real work happens in child processes
    or containers, so the GIL is not a bottleneck.
    """
    if not items:
        return []
    workers = max(1, min(max_workers or default_concurrency(), len(items)))
    start = time.monotonic()

    def one(item: tuple[str, Problem]) -> ExecutionResult:
        solution, problem = item
        try:
            return executor.run(solution, problem)
        except Exception as exc:  # noqa: BLE001 - isolate every item
            log.error(
                "sandbox_item_crashed",
                task_id=problem.task_id,
                backend=executor.name,
                error=repr(exc),
            )
            return ExecutionResult.sandbox_error(
                len(problem.tests), f"executor crashed: {exc!r}", backend=executor.name
            )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sandbox") as pool:
        results = list(pool.map(one, items))

    elapsed = time.monotonic() - start
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    log.info(
        "sandbox_batch_done",
        backend=executor.name,
        n=len(items),
        workers=workers,
        seconds=round(elapsed, 3),
        per_second=round(len(items) / elapsed, 1) if elapsed > 0 else None,
        statuses=by_status,
    )
    return results
