"""Docker backend: a warm pool of locked-down, long-lived containers (default on macOS).

``docker run`` per job costs ~0.5s, so each container instead runs ``server.py``, which
executes jobs one at a time, each in a FRESH child interpreter with rlimits, and wipes the
container between jobs. Containers are recycled after ``jobs_per_container`` jobs or whenever
they misbehave (no answer in time, bad protocol, dead process).

Every container: ``--network none --read-only --tmpfs /tmp --memory/--memory-swap --cpus 1
--pids-limit 64 --cap-drop ALL --security-opt no-new-privileges --ipc none --init``, non-root
user, no host environment, code passed over stdin (never via files or mounts). Containers are
labelled ``coderl.sandbox=1`` and removed by label on shutdown and on the next start-up.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import secrets
import selectors
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from coderl.sandbox import metrics
from coderl.sandbox.backends.base import Executor, SandboxUnavailableError
from coderl.sandbox.types import SandboxLimits
from coderl.utils.logging import get_logger

log = get_logger(__name__)

SANDBOX_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SANDBOX_DIR.parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "sandbox.Dockerfile"
IMAGE_FILES = ("harness.py", "runner.py", "server.py")
IMAGE_REPO = "coderl-sandbox"
LABEL = "coderl.sandbox"
CONTAINER_MEMORY_HEADROOM_MB = 64  # in-container server + tmpfs, on top of the job's RLIMIT_AS
TMPFS_MB = 64
PIDS_LIMIT = 64
STARTUP_TIMEOUT_S = 60.0
RESPONSE_SLACK_S = 20.0


class ContainerError(RuntimeError):
    """A pool container misbehaved; it is discarded."""


# ---------------------------------------------------------------------- docker helpers
def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def image_tag() -> str:
    """Content-addressed tag: any change to the Dockerfile or harness forces a rebuild."""
    h = hashlib.sha256(DOCKERFILE.read_bytes())
    for name in IMAGE_FILES:
        h.update(name.encode())
        h.update((SANDBOX_DIR / name).read_bytes())
    return f"{IMAGE_REPO}:{h.hexdigest()[:12]}"


def image_exists(tag: str) -> bool:
    proc = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, timeout=30)
    return proc.returncode == 0


def build_image(tag: str | None = None) -> str:
    tag = tag or image_tag()
    log.info("sandbox_image_build", tag=tag)
    proc = subprocess.run(
        [
            "docker",
            "build",
            "--quiet",
            "-f",
            str(DOCKERFILE),
            "-t",
            tag,
            "-t",
            f"{IMAGE_REPO}:latest",
            "--label",
            f"{LABEL}=1",
            str(SANDBOX_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise SandboxUnavailableError(f"docker build failed:\n{proc.stderr[-2000:]}")
    return tag


def ensure_image(build: bool = True) -> str:
    tag = image_tag()
    if image_exists(tag):
        return tag
    if not build:
        raise SandboxUnavailableError(f"sandbox image {tag} missing; run `make sandbox-image`")
    return build_image(tag)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def list_containers(extra_filter: str | None = None) -> list[dict[str, str]]:
    cmd = ["docker", "ps", "-a", "--filter", f"label={LABEL}=1"]
    if extra_filter:
        cmd += ["--filter", extra_filter]
    cmd += ["--format", "{{json .}}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    rows = []
    for line in proc.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _labels(row: dict[str, str]) -> dict[str, str]:
    out = {}
    for kv in row.get("Labels", "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def remove_containers(ids: list[str]) -> None:
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=120)


def cleanup_containers(all_owners: bool = False) -> int:
    """Remove sandbox containers whose owning process is gone (or all of them)."""
    host = socket.gethostname()
    stale = []
    for row in list_containers():
        labels = _labels(row)
        try:
            owner_pid = int(labels.get("coderl.owner_pid", "0"))
        except ValueError:
            owner_pid = 0
        same_host = labels.get("coderl.owner_host") == host
        if all_owners or (same_host and not _pid_alive(owner_pid)):
            stale.append(row["ID"])
    remove_containers(stale)
    if stale:
        log.info("sandbox_containers_removed", n=len(stale))
    return len(stale)


# ---------------------------------------------------------------------- one container
class _Container:
    def __init__(self, name: str, argv: list[str]) -> None:
        self.name = name
        self.jobs = 0
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Ctrl-C in the terminal must not kill the pool mid-job
        )
        assert self.proc.stdout is not None
        self._out = self.proc.stdout.fileno()
        os.set_blocking(self._out, False)
        self._buf = bytearray()
        hello = self._read_line(time.monotonic() + STARTUP_TIMEOUT_S)
        if not hello.get("ready"):
            raise ContainerError(f"{name}: bad hello {hello!r}")

    def _read_line(self, deadline: float) -> dict[str, Any]:
        sel = selectors.DefaultSelector()
        sel.register(self._out, selectors.EVENT_READ)
        try:
            while True:
                nl = self._buf.find(b"\n")
                if nl >= 0:
                    line = bytes(self._buf[:nl])
                    del self._buf[: nl + 1]
                    try:
                        obj = json.loads(line)
                    except ValueError as exc:
                        raise ContainerError(f"{self.name}: non-JSON output") from exc
                    if not isinstance(obj, dict):
                        raise ContainerError(f"{self.name}: unexpected output")
                    return obj
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ContainerError(f"{self.name}: no response in time")
                if not sel.select(timeout=remaining):
                    continue
                try:
                    chunk = os.read(self._out, 1 << 20)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise ContainerError(f"{self.name}: container exited")
                self._buf += chunk
                if len(self._buf) > 64 * 1024 * 1024:
                    raise ContainerError(f"{self.name}: oversized response")
        finally:
            sel.close()

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        token = secrets.token_hex(16)
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps({**payload, "token": token}).encode() + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ContainerError(f"{self.name}: stdin closed") from exc
        resp = self._read_line(time.monotonic() + timeout)
        if resp.get("token") != token:
            raise ContainerError(f"{self.name}: response token mismatch")
        return resp

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except OSError:
            pass
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if self.proc.stdout is not None:
            self.proc.stdout.close()


# ---------------------------------------------------------------------- backend
class DockerBackend(Executor):
    name = "docker"

    def __init__(
        self,
        limits: SandboxLimits | None = None,
        *,
        pool_size: int | None = None,
        jobs_per_container: int = 1000,
        build: bool = True,
        warm: bool = True,
    ) -> None:
        super().__init__(limits)
        if not docker_available():
            raise SandboxUnavailableError("docker is not installed or the daemon is not running")
        self.image = ensure_image(build=build)
        self.pool_size = max(1, pool_size or (os.cpu_count() or 1))
        self.jobs_per_container = jobs_per_container
        self.session = secrets.token_hex(6)
        self._free: queue.LifoQueue[_Container] = queue.LifoQueue()
        self._lock = threading.Lock()
        self._live = 0
        self._counter = 0
        self._closed = False
        cleanup_containers()
        atexit.register(self.close)
        if warm:
            self.warm()
        log.info(
            "sandbox_backend_ready",
            backend=self.name,
            image=self.image,
            pool_size=self.pool_size,
            session=self.session,
        )

    @property
    def default_concurrency(self) -> int:
        return self.pool_size

    # ------------------------------------------------------------------ pool management
    def _docker_argv(self, name: str) -> list[str]:
        lim = self.limits
        mem = f"{lim.memory_mb + CONTAINER_MEMORY_HEADROOM_MB}m"
        return [
            "docker",
            "run",
            "-i",
            "--rm",
            "--init",
            "--name",
            name,
            "--label",
            f"{LABEL}=1",
            "--label",
            f"coderl.session={self.session}",
            "--label",
            f"coderl.owner_pid={os.getpid()}",
            "--label",
            f"coderl.owner_host={socket.gethostname()}",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={TMPFS_MB}m,mode=1777",
            "--ipc",
            "none",
            "--memory",
            mem,
            "--memory-swap",
            mem,
            "--cpus",
            "1",
            "--pids-limit",
            str(PIDS_LIMIT),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "10001:10001",
            "--hostname",
            "sandbox",
            "--log-driver",
            "none",
            "--pull",
            "never",
            self.image,
        ]

    def _start_container(self) -> _Container:
        with self._lock:
            self._counter += 1
            name = f"coderl-sbx-{self.session}-{self._counter}"
        return _Container(name, self._docker_argv(name))

    def warm(self) -> None:
        """Start containers until the pool is full (in parallel)."""
        with self._lock:
            missing = self.pool_size - self._live
            self._live += max(0, missing)
        if missing <= 0:
            return

        def start(_: int) -> _Container | None:
            try:
                return self._start_container()
            except Exception as exc:  # noqa: BLE001
                log.error("sandbox_container_start_failed", error=repr(exc))
                return None

        with ThreadPoolExecutor(max_workers=missing) as ex:
            started = list(ex.map(start, range(missing)))
        failed = 0
        for c in started:
            if c is None:
                failed += 1
            else:
                self._free.put(c)
        if failed:
            with self._lock:
                self._live -= failed
            if failed == missing:
                raise SandboxUnavailableError("could not start any sandbox container")

    def _acquire(self) -> _Container:
        while True:
            try:
                c = self._free.get_nowait()
            except queue.Empty:
                with self._lock:
                    can_start = self._live < self.pool_size
                    if can_start:
                        self._live += 1
                if can_start:
                    try:
                        return self._start_container()
                    except Exception:
                        with self._lock:
                            self._live -= 1
                        raise
                c = self._free.get()
            if c.alive():
                return c
            self._discard(c, "dead")

    def _discard(self, c: _Container, reason: str) -> None:
        metrics.CONTAINER_RECYCLES.labels(reason).inc()
        log.info("sandbox_container_recycled", container=c.name, reason=reason, jobs=c.jobs)
        with self._lock:
            self._live -= 1
        threading.Thread(target=c.kill, daemon=True).start()

    def _release(self, c: _Container, healthy: bool) -> None:
        if self._closed:
            c.kill()
        elif not healthy:
            self._discard(c, "misbehaved")
        elif c.jobs >= self.jobs_per_container:
            self._discard(c, "max_jobs")
        else:
            self._free.put(c)

    # ------------------------------------------------------------------ execution
    def _execute(self, job: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise SandboxUnavailableError("backend is closed")
        c = self._acquire()
        healthy = False
        try:
            resp = c.request(
                {"op": "run", "job": job, "limits": self.limits.wire()},
                timeout=self.limits.wall_seconds + RESPONSE_SLACK_S,
            )
            c.jobs += 1
            result = resp.get("result")
            if not isinstance(result, dict):
                raise ContainerError(f"{c.name}: malformed result")
            healthy = result.get("status") != "sandbox_error"
            return result
        finally:
            self._release(c, healthy)

    def stray_processes(self) -> dict[str, list[int]]:
        """Processes left in idle containers besides tini/server/spare (should be none)."""
        out: dict[str, list[int]] = {}
        held = []
        while True:
            try:
                held.append(self._free.get_nowait())
            except queue.Empty:
                break
        try:
            for c in held:
                out[c.name] = list(c.request({"op": "ps"}, timeout=30).get("stray_pids", []))
        finally:
            for c in held:
                self._free.put(c)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        containers = []
        while True:
            try:
                containers.append(self._free.get_nowait())
            except queue.Empty:
                break
        for c in containers:
            try:
                assert c.proc.stdin is not None
                c.proc.stdin.write(b'{"op": "exit"}\n')
                c.proc.stdin.flush()
            except (OSError, AssertionError):
                pass
        # Remove everything from this session by label, including busy/leaked containers.
        rows = list_containers(f"label=coderl.session={self.session}")
        remove_containers([r["ID"] for r in rows])
        for c in containers:
            try:
                c.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                c.proc.kill()
            if c.proc.stdout is not None:
                c.proc.stdout.close()
            if c.proc.stdin is not None:
                try:
                    c.proc.stdin.close()
                except OSError:
                    pass
        log.info("sandbox_backend_closed", backend=self.name, session=self.session)
