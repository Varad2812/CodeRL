"""Shared fixtures for the sandbox test-suite.

Backends are parametrized: the subprocess backend runs on Linux only, the docker backend
whenever a Docker daemon is reachable (tests marked ``docker``). Each backend comes in two
flavours: ``executor`` (production configuration) and ``os_only_executor`` (the in-process
audit hook disabled), which proves the OS-level layers hold on their own.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pytest

from coderl.data.schema import Problem
from coderl.sandbox import ExecutionResult, Executor, SandboxLimits

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_MARKER = "coderl-sandbox-harness"


@lru_cache(maxsize=1)
def docker_ok() -> bool:
    from coderl.sandbox.backends.docker_backend import docker_available

    return docker_available()


BACKEND_PARAMS = [
    pytest.param(
        "subprocess",
        marks=pytest.mark.skipif(
            sys.platform != "linux", reason="subprocess backend is Linux-only"
        ),
        id="subprocess",
    ),
    pytest.param(
        "docker",
        marks=[
            pytest.mark.docker,
            pytest.mark.skipif(not docker_ok(), reason="docker daemon not available"),
        ],
        id="docker",
    ),
]


def _make(name: str, limits: SandboxLimits) -> Executor:
    if name == "docker":
        from coderl.sandbox.backends.docker_backend import DockerBackend

        return DockerBackend(limits, pool_size=2)
    from coderl.sandbox.backends.subprocess_backend import SubprocessBackend

    return SubprocessBackend(limits)


@pytest.fixture(scope="session", params=BACKEND_PARAMS)
def executor(request: pytest.FixtureRequest) -> Iterator[Executor]:
    ex = _make(request.param, SandboxLimits())
    yield ex
    ex.close()


@pytest.fixture(scope="session", params=BACKEND_PARAMS)
def os_only_executor(request: pytest.FixtureRequest) -> Iterator[Executor]:
    """Audit hook disabled: only rlimits / namespaces / container flags protect the host."""
    ex = _make(request.param, SandboxLimits(audit_hook=False))
    if request.param == "subprocess":
        isolation = getattr(ex, "isolation", "none")
        if isolation != "bwrap":
            ex.close()
            pytest.skip(f"OS-level isolation is {isolation!r}; install bubblewrap for these tests")
    yield ex
    ex.close()


def problem(tests: list[str], entry_point: str = "f", test_setup: str = "") -> Problem:
    return Problem(
        task_id="sandbox/test",
        source="mbpp",
        split="test",
        prompt="",
        entry_point=entry_point,
        tests=tests,
        test_setup=test_setup,
    )


def harness_processes() -> list[int]:
    """PIDs of harness interpreters on this (Linux) host."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if HARNESS_MARKER.encode() in cmdline and str(os.getpid()) != entry.name:
            pids.append(int(entry.name))
    return pids


def assert_no_orphans(ex: Executor) -> None:
    if ex.name == "docker":
        stray = ex.stray_processes()  # type: ignore[attr-defined]
        assert all(not pids for pids in stray.values()), stray
    else:
        assert harness_processes() == []


def assert_scored_zero(r: ExecutionResult) -> None:
    assert r.num_passed == 0, r
    assert r.status != "passed"
    assert all(not t.passed for t in r.tests)
