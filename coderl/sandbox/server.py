"""In-container job supervisor for the Docker backend. STDLIB ONLY.

Runs as the container's main process (under ``tini``), as a non-root user, and never executes
untrusted code itself. It reads one JSON request per line on stdin and answers with one JSON
line on stdout. Each job runs in a FRESH harness interpreter (``runner.spawn_harness``) with
rlimits; afterwards every other process in the container is killed and /tmp is wiped, so jobs
cannot contaminate each other. One spare harness is pre-started (after cleanup, so no job ever
coexists with it) to hide interpreter start-up latency.

Requests:  {"op": "run", "token": str, "job": {...}, "limits": {...}}  -> {"token", "result"}
           {"op": "ps",  "token": str}  -> {"token", "stray_pids": [...]}  (for tests)
           {"op": "exit"}
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "harness.py")
JOB_ROOT = "/tmp"

try:  # inside the package (tests on the host)
    from coderl.sandbox import runner
except ImportError:  # inside the container: sibling module, -I removed the script dir
    sys.path.insert(0, HERE)
    import runner  # type: ignore[import-not-found,no-redef]


def make_nondumpable() -> None:
    """Hide /proc/<server>/{fd,mem,environ} from the (same-uid) job processes."""
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
    except Exception:  # noqa: BLE001 - best effort; the audit hook also blocks /proc
        pass


def stray_pids(keep: set[int]) -> list[int]:
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            pid = int(name)
            if pid not in keep:
                pids.append(pid)
    return pids


def sweep(keep: set[int]) -> None:
    """Kill every process except tini, ourselves and `keep`; wipe the shared temp dirs."""
    for _ in range(3):
        pids = stray_pids(keep)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        _reap_all()
    for root in (JOB_ROOT, "/dev/shm"):
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        for name in entries:
            path = os.path.join(root, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
            except OSError:
                pass


def _reap_all() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


class Spare:
    def __init__(self) -> None:
        self.hp: runner.HarnessProcess | None = None
        self.key: str | None = None
        self.cwd: str | None = None

    def take(self, limits: dict[str, Any]) -> tuple[runner.HarnessProcess, str]:
        key = json.dumps(limits, sort_keys=True)
        if self.hp is not None and self.key == key and self.hp.proc.poll() is None:
            hp, cwd = self.hp, self.cwd
            self.hp = None
            assert cwd is not None
            return hp, cwd
        self.discard()
        return spawn(limits)

    def refill(self, limits: dict[str, Any]) -> None:
        self.discard()
        self.hp, self.cwd = spawn(limits)
        self.key = json.dumps(limits, sort_keys=True)

    def discard(self) -> None:
        if self.hp is not None:
            self.hp.close()
        self.hp = None

    def pids(self) -> set[int]:
        return {self.hp.pid} if self.hp is not None else set()


def spawn(limits: dict[str, Any]) -> tuple[runner.HarnessProcess, str]:
    cwd = tempfile.mkdtemp(prefix="job-", dir=JOB_ROOT)
    hp = runner.spawn_harness(
        [sys.executable, "-I", "-S", "-u", HARNESS],
        limits,
        cwd=cwd,
        nproc=int(limits["max_processes"]),
    )
    return hp, cwd


def respond(obj: dict[str, Any]) -> None:
    sys.stdout.buffer.write(json.dumps(obj).encode() + b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    make_nondumpable()
    me = os.getpid()
    base_keep = {1, me}
    spare = Spare()
    respond({"ready": True, "python": sys.version.split()[0], "pid": me})
    for raw in sys.stdin.buffer:
        try:
            req = json.loads(raw)
        except ValueError:
            continue
        op = req.get("op")
        token = req.get("token")
        if op == "run":
            limits = req["limits"]
            hp, _cwd = spare.take(limits)
            try:
                result = runner.execute(hp, req["job"], limits)
            except Exception as exc:  # noqa: BLE001 - report, never crash the loop
                result = {"status": "sandbox_error", "error": repr(exc)}
            finally:
                hp.close()
            sweep(base_keep)
            spare.refill(limits)  # started now, boots while the host handles the response
            respond({"token": token, "result": result})
        elif op == "ps":
            respond({"token": token, "stray_pids": stray_pids(base_keep | spare.pids())})
        elif op == "exit":
            break
    spare.discard()


if __name__ == "__main__":
    main()
