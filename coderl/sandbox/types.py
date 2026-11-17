"""Result and configuration types for the sandbox executor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal[
    "passed",
    "failed",
    "syntax_error",
    "runtime_error",
    "timeout",
    "memory_exceeded",
    "sandbox_error",
]

MAX_MESSAGE_CHARS = 500
MAX_TRACEBACK_CHARS = 2000


class SandboxLimits(BaseModel):
    """Per-job resource limits. Enforced by the OS (rlimits / cgroups), not by Python."""

    cpu_seconds: int = 2  # RLIMIT_CPU of the job process
    wall_seconds: float = 4.0  # whole job, enforced by the trusted parent (kills process group)
    per_test_seconds: float = 1.0  # per-test budget inside the harness (SIGALRM)
    memory_mb: int = 256  # RLIMIT_AS of the job process (Docker: cgroup limit adds headroom)
    max_output_bytes: int = 64 * 1024  # stdout/stderr tail kept per stream
    max_file_mb: int = 16  # RLIMIT_FSIZE
    max_open_files: int = 64  # RLIMIT_NOFILE
    max_processes: int = 16  # RLIMIT_NPROC headroom above the sandbox user's baseline
    # In-process hardening (audit hook). Only ever disabled by the security test-suite, to prove
    # that the OS-level layers hold on their own.
    audit_hook: bool = True

    def wire(self) -> dict[str, Any]:
        """Plain-dict form understood by the stdlib-only runner (inside Docker too)."""
        return self.model_dump()


class TestResult(BaseModel):
    __test__ = False  # not a pytest test class

    index: int
    passed: bool
    error_type: str | None = None
    message: str = ""  # truncated to MAX_MESSAGE_CHARS
    # e.g. "custom_eq": the value under test came from a class that overrides __eq__/__bool__
    flags: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    status: Status
    tests: list[TestResult] = Field(default_factory=list)
    num_passed: int = 0
    num_total: int = 0
    error_type: str | None = None
    error_lineno: int | None = None  # line in the solution (syntax errors, module-level errors)
    # Truncated (~2000 chars) traceback containing ONLY frames from the solution/tests;
    # harness-internal frames are stripped. Intended to be shown to the model for self-repair.
    traceback: str = ""
    stdout: str = ""  # tail, truncated to max_output_bytes
    stderr: str = ""
    duration_ms: float = 0.0
    # Integrity signals, e.g. "forged_records" (writes to the result channel without the nonce),
    # "custom_eq", "builtins_modified", "modules_modified".
    flags: list[str] = Field(default_factory=list)
    backend: str = ""

    @property
    def pass_fraction(self) -> float:
        return self.num_passed / self.num_total if self.num_total else 0.0

    @classmethod
    def sandbox_error(
        cls, num_total: int, message: str, backend: str = "", duration_ms: float = 0.0
    ) -> ExecutionResult:
        """Infrastructure failure: every test counts as failed (never as passed)."""
        return cls(
            status="sandbox_error",
            tests=[
                TestResult(index=i, passed=False, error_type="SandboxError", message=message)
                for i in range(num_total)
            ],
            num_passed=0,
            num_total=num_total,
            error_type="SandboxError",
            traceback=message[:MAX_TRACEBACK_CHARS],
            duration_ms=duration_ms,
            backend=backend,
        )
