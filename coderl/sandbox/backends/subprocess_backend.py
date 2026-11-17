"""Subprocess backend: a fresh, rlimited, namespaced interpreter per job (LINUX ONLY).

Intended for GPU training boxes and CI, where Docker per job is too heavy. Isolation tiers,
detected once at startup (best first):

* ``bwrap``   bubblewrap: new user/pid/net/ipc/uts namespaces, read-only root filesystem,
              $HOME / repo / data dirs hidden behind read-only tmpfs, private /proc, fresh
              /tmp per job, killed with its parent.
* ``unshare`` util-linux unshare: new user + net (+ pid/mount with private /proc when
              possible). Network is isolated, the filesystem is NOT.
* ``none``    rlimits and the in-process audit hook only. Logged as a WARNING.

On macOS the backend refuses to start (RLIMIT_AS and network namespaces do not exist there)
unless ``CODERL_ALLOW_UNSAFE_SUBPROCESS=1`` is set for quick local development.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from coderl.sandbox import runner
from coderl.sandbox.backends.base import Executor, SandboxUnavailableError
from coderl.sandbox.types import SandboxLimits
from coderl.utils.logging import get_logger

log = get_logger(__name__)

HARNESS_PATH = Path(__file__).resolve().parent.parent / "harness.py"
UNSAFE_ENV = "CODERL_ALLOW_UNSAFE_SUBPROCESS"

UNSHARE_FULL = [
    "unshare",
    "--user",
    "--map-root-user",
    "--net",
    "--pid",
    "--fork",
    "--mount",
    "--mount-proc",
    "--kill-child",
]
UNSHARE_NET = ["unshare", "--user", "--map-root-user", "--net"]


def base_python() -> tuple[str, str]:
    """(interpreter binary, installation prefix), bypassing any virtualenv."""
    exe = getattr(sys, "_base_executable", None) or sys.executable
    return os.path.realpath(exe), os.path.realpath(sys.base_prefix)


def _works(cmd: list[str]) -> bool:
    try:
        proc = subprocess.run(
            [*cmd, "true"], capture_output=True, timeout=10, env={"PATH": "/usr/bin:/bin"}
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


@lru_cache(maxsize=1)
def detect_isolation() -> tuple[str, tuple[str, ...]]:
    """Return (tier, unshare argv prefix) for this machine. Cached per process."""
    if sys.platform != "linux":
        return "none", ()
    # probe with the same namespaces and mounts real jobs use (e.g. a fresh /proc can be
    # refused inside containers that mask /proc paths)
    probe = [
        "bwrap",
        "--die-with-parent",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--remount-ro",
        "/tmp",
        "--",
    ]
    if shutil.which("bwrap") and _works(probe):
        return "bwrap", ()
    if shutil.which("unshare"):
        for cmd in (UNSHARE_FULL, UNSHARE_NET):
            if _works([*cmd, "--"]):
                return "unshare", tuple(cmd)
    return "none", ()


def _hidden_dirs() -> list[str]:
    """Host directories the sandbox must not see (secrets, repo, datasets)."""
    candidates = [
        os.path.expanduser("~"),
        os.getcwd(),
        os.environ.get("CODERL_DATA_DIR", ""),
        "/root",
        "/workspace",
        "/mnt",
        "/media",
        "/srv",
        "/opt",
        "/var/lib",
        "/var/tmp",
        "/run",
    ]
    out: list[str] = []
    for c in candidates:
        if not c:
            continue
        p = os.path.realpath(c)
        if p in ("/", "/usr", "/bin", "/lib", "/etc") or not os.path.isdir(p):
            continue
        if any(p == o or p.startswith(o + "/") for o in out):
            continue
        out = [o for o in out if not o.startswith(p + "/")]
        out.append(p)
    return out


_task_cache: dict[str, Any] = {"t": 0.0, "n": 0}


def _user_task_count() -> int:
    """Tasks (threads) owned by our uid: RLIMIT_NPROC counts these globally. Cached 1s."""
    now = time.monotonic()
    if now - _task_cache["t"] < 1.0:
        return int(_task_cache["n"])
    uid = os.getuid()
    n = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            n += len(os.listdir(f"/proc/{entry.name}/task"))
        except OSError:
            continue
    _task_cache.update(t=now, n=n)
    return n


class SubprocessBackend(Executor):
    name = "subprocess"

    def __init__(self, limits: SandboxLimits | None = None) -> None:
        super().__init__(limits)
        if sys.platform != "linux":
            if os.environ.get(UNSAFE_ENV) != "1":
                raise SandboxUnavailableError(
                    "the subprocess sandbox backend is Linux-only: macOS has no working "
                    "RLIMIT_AS and no network/user namespaces, so untrusted code would run "
                    "with network and filesystem access. Use CODERL_SANDBOX_BACKEND=docker, "
                    f"or set {UNSAFE_ENV}=1 for quick local development with TRUSTED code."
                )
            log.warning(
                "sandbox_unsafe_subprocess",
                reason=f"{UNSAFE_ENV}=1 on {sys.platform}: no network or filesystem isolation",
            )
        self.isolation, self._unshare = detect_isolation()
        self._python, self._prefix = base_python()
        self._harness_src = HARNESS_PATH.read_text()
        self._hidden = _hidden_dirs()
        self._root = tempfile.mkdtemp(prefix="coderl-sandbox-")
        if self.isolation != "bwrap" and sys.platform == "linux":
            log.warning(
                "sandbox_weak_isolation",
                backend=self.name,
                isolation=self.isolation,
                hint="install bubblewrap (apt install bubblewrap) for filesystem isolation"
                + ("" if self.isolation == "unshare" else " and network isolation"),
            )
        log.info("sandbox_backend_ready", backend=self.name, isolation=self.isolation)

    # ------------------------------------------------------------------ command line
    def _harness_argv(self) -> list[str]:
        # -c: the harness source travels in argv, so the repo can stay hidden from the sandbox
        return [self._python, "-I", "-S", "-u", "-c", self._harness_src]

    def _bwrap(self, jobdir: str) -> list[str]:
        cmd = [
            "bwrap",
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
        ]
        for d in self._hidden:
            cmd += ["--tmpfs", d]
        # the interpreter may live under a hidden dir (uv installs it in ~/.local/share/uv)
        cmd += ["--ro-bind", self._prefix, self._prefix]
        for d in self._hidden:
            cmd += ["--remount-ro", d]
        cmd += ["--bind", jobdir, "/tmp", "--chdir", "/tmp", "--"]
        return cmd

    def _nproc(self) -> int:
        if sys.platform != "linux":
            return 0
        if self.isolation == "bwrap" or "--user" in self._unshare:
            # fresh user namespace: RLIMIT_NPROC is accounted per namespace (kernel >= 5.14)
            return self.limits.max_processes
        return _user_task_count() + self.limits.max_processes

    # ------------------------------------------------------------------ execution
    def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        jobdir = tempfile.mkdtemp(prefix="job-", dir=self._root)
        try:
            if self.isolation == "bwrap":
                wrapper, cwd = self._bwrap(jobdir), None
            else:
                wrapper, cwd = [*self._unshare, "--"] if self._unshare else [], jobdir
            hp = runner.spawn_harness(
                self._harness_argv(),
                self.limits.wire(),
                cwd=cwd,
                nproc=self._nproc(),
                wrapper=wrapper,
                signal_via_exit_code=bool(wrapper),
            )
            return runner.execute(hp, job, self.limits.wire())
        finally:
            shutil.rmtree(jobdir, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
