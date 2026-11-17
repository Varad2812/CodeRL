"""Sandbox backends and backend selection."""

from __future__ import annotations

import os
import sys

from coderl.sandbox.backends.base import Executor, SandboxUnavailableError
from coderl.sandbox.types import SandboxLimits
from coderl.utils.logging import get_logger

log = get_logger(__name__)

BACKEND_ENV = "CODERL_SANDBOX_BACKEND"
BACKENDS = ("subprocess", "docker")


def default_backend_name() -> str:
    """docker on macOS/Windows (no usable namespaces/RLIMIT_AS), subprocess on Linux."""
    return "subprocess" if sys.platform == "linux" else "docker"


def get_executor(
    backend: str | None = None, limits: SandboxLimits | None = None, **kwargs: object
) -> Executor:
    """Create the executor named by ``backend``, ``$CODERL_SANDBOX_BACKEND`` or the default."""
    requested = backend or os.environ.get(BACKEND_ENV) or ""
    name = requested.strip().lower() or default_backend_name()
    if name not in BACKENDS:
        raise ValueError(f"unknown sandbox backend {name!r}; expected one of {BACKENDS}")
    source = "argument" if backend else ("env" if os.environ.get(BACKEND_ENV) else "default")
    log.info("sandbox_backend_selected", backend=name, source=source, platform=sys.platform)
    if name == "docker":
        from coderl.sandbox.backends.docker_backend import DockerBackend

        return DockerBackend(limits, **kwargs)  # type: ignore[arg-type]
    from coderl.sandbox.backends.subprocess_backend import SubprocessBackend

    return SubprocessBackend(limits, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "BACKENDS",
    "BACKEND_ENV",
    "Executor",
    "SandboxUnavailableError",
    "default_backend_name",
    "get_executor",
]
