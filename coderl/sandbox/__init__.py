"""Secure execution of untrusted, model-generated code against unit tests.

>>> from coderl.sandbox import get_executor
>>> with get_executor() as ex:              # docker on macOS, subprocess on Linux
...     result = ex.run(solution, problem)  # -> ExecutionResult
...     results = ex.run_batch([(solution, problem), ...])

See SECURITY.md for the threat model and the guarantees of each backend.
"""

from coderl.sandbox.backends import (
    BACKEND_ENV,
    Executor,
    SandboxUnavailableError,
    default_backend_name,
    get_executor,
)
from coderl.sandbox.types import ExecutionResult, SandboxLimits, TestResult

__all__ = [
    "BACKEND_ENV",
    "ExecutionResult",
    "Executor",
    "SandboxLimits",
    "SandboxUnavailableError",
    "TestResult",
    "default_backend_name",
    "get_executor",
]
