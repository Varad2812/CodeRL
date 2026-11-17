"""Trusted parent side of the harness protocol. STDLIB ONLY.

Used directly by the subprocess backend (on the host) and by ``server.py`` inside each Docker
container. It never executes untrusted code itself: it spawns a fresh harness interpreter
per job, feeds it the job in two authenticated messages, collects nonce-tagged records from a
dedicated pipe, enforces the wall-clock limit by killing the whole process group, and turns
everything into a plain-dict result. Every test starts as FAILED and only an authenticated
"T ... passed" record from the harness can flip it.
"""

from __future__ import annotations

import hmac
import marshal
import os
import secrets
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

MAX_MESSAGE_CHARS = 500
MAX_TRACEBACK_CHARS = 2000
MAX_RESULT_CHANNEL_BYTES = 4 * 1024 * 1024
DRAIN_GRACE_SECONDS = 0.25

SCRUBBED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HOME": "/tmp",
    "PYTHONHASHSEED": "0",
}


def _frame(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "big") + payload


def _decode(field_hex: bytes) -> str:
    try:
        return bytes.fromhex(field_hex.decode("ascii")).decode("utf-8", "replace")
    except (ValueError, UnicodeDecodeError):
        return ""


def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _tail(buf: bytearray, limit: int) -> str:
    data = bytes(buf[-limit:]) if len(buf) > limit else bytes(buf)
    return data.decode("utf-8", "replace")


def _reap(proc: subprocess.Popen[bytes]) -> float:
    """Wait for the (already SIGKILLed) process; return the CPU seconds it used."""
    if proc.returncode is not None:
        return 0.0
    try:
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        return float(usage.ru_utime + usage.ru_stime)
    except ChildProcessError:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return 0.0


@dataclass
class HarnessProcess:
    """A freshly started harness interpreter, waiting for its job on stdin."""

    proc: subprocess.Popen[bytes]
    result_r: int
    signal_via_exit_code: bool = False
    started: float = field(default_factory=time.monotonic)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def kill(self) -> None:
        """SIGKILL the whole process group (the harness is a session leader)."""
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass

    def close(self) -> None:
        self.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
        try:
            os.close(self.result_r)
        except OSError:
            pass


def spawn_harness(
    harness_argv: list[str],
    limits: dict[str, Any],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    nproc: int = 0,
    wrapper: list[str] | None = None,
    signal_via_exit_code: bool = False,
) -> HarnessProcess:
    """Start ``[*wrapper, *harness_argv, <result_fd>, <rlimits>]`` in a new session.

    rlimits are applied by the harness itself before it reads any input (no preexec_fn, which
    is unsafe in a multi-threaded parent); untrusted code cannot raise them afterwards.
    """
    result_r, result_w = os.pipe()
    os.set_inheritable(result_w, True)
    rl = ",".join(
        str(int(v))
        for v in (
            limits["cpu_seconds"],
            limits["memory_mb"],
            limits["max_file_mb"],
            limits["max_open_files"],
            nproc,
        )
    )
    argv = [*(wrapper or []), *harness_argv, str(result_w), rl]
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=dict(SCRUBBED_ENV if env is None else env),
            pass_fds=(result_w,),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(result_w)
    return HarnessProcess(proc=proc, result_r=result_r, signal_via_exit_code=signal_via_exit_code)


def _new_result(num_total: int) -> dict[str, Any]:
    return {
        "status": "runtime_error",
        "tests": [
            {
                "index": i,
                "passed": False,
                "error_type": "NotRun",
                "message": "test did not run",
                "flags": [],
            }
            for i in range(num_total)
        ],
        "num_passed": 0,
        "num_total": num_total,
        "error_type": None,
        "error_lineno": None,
        "traceback": "",
        "stdout": "",
        "stderr": "",
        "duration_ms": 0.0,
        "flags": [],
    }


def execute(hp: HarnessProcess, job: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """Run one job on a fresh harness process and return a plain-dict ExecutionResult."""
    tests = list(job["tests"])
    result = _new_result(len(tests))
    start = time.monotonic()
    deadline = start + float(limits["wall_seconds"])
    cap = int(limits["max_output_bytes"])
    hard_output_cap = cap * 16
    nonce = secrets.token_hex(16)
    nonce_b = nonce.encode("ascii")

    m1 = _frame(
        marshal.dumps(
            (
                nonce,
                job["solution"],
                job["entry_point"],
                tuple(job.get("protect_modules", ())),
                float(limits["per_test_seconds"]),
                bool(limits.get("audit_hook", True)),
            )
        )
    )
    m2 = _frame(marshal.dumps((nonce, job.get("test_setup", ""), tuple(tests))))

    proc = hp.proc
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    stdin_fd = proc.stdin.fileno()
    os.set_blocking(stdin_fd, False)
    out = {proc.stdout.fileno(): bytearray(), proc.stderr.fileno(): bytearray()}
    out_total = 0
    res_buf = bytearray()
    res_total = 0
    pending = bytearray(m1)
    stdin_open = True

    sel = selectors.DefaultSelector()
    sel.register(stdin_fd, selectors.EVENT_WRITE)
    for fd in (*out, hp.result_r):
        sel.register(fd, selectors.EVENT_READ)
    open_reads = {*out, hp.result_r}

    phase = "load"  # load -> tests -> finished
    terminal: str | None = None  # S / E / D
    killed_by_us: str | None = None  # "timeout" | "output" | "protocol"
    forged = 0
    test_records: dict[int, dict[str, Any]] = {}
    harness_flags: list[str] = []
    finish_at: float | None = None

    def close_stdin() -> None:
        nonlocal stdin_open
        if stdin_open:
            stdin_open = False
            try:
                sel.unregister(stdin_fd)
            except (KeyError, ValueError):
                pass
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

    def handle_record(line: bytes) -> None:
        nonlocal forged, phase, terminal, pending, finish_at
        parts = line.split(b"\t")
        if len(parts) < 2 or not hmac.compare_digest(parts[0], nonce_b):
            forged += 1
            return
        kind = parts[1]
        fields = [_decode(p) for p in parts[2:]]
        if kind == b"L" and phase == "load":
            phase = "tests"
            pending = bytearray(m2)
            if stdin_open:
                try:
                    sel.modify(stdin_fd, selectors.EVENT_WRITE)
                except (KeyError, ValueError):
                    sel.register(stdin_fd, selectors.EVENT_WRITE)
        elif kind in (b"S", b"E") and phase == "load" and len(fields) >= 4:
            terminal = kind.decode()
            phase = "finished"
            result["error_type"] = fields[0] or None
            result["traceback"] = _clip(fields[2], MAX_TRACEBACK_CHARS)
            try:
                result["error_lineno"] = int(fields[3]) or None
            except ValueError:
                result["error_lineno"] = None
            result["_message"] = fields[1]
            finish_at = time.monotonic() + DRAIN_GRACE_SECONDS
        elif kind == b"T" and phase == "tests" and len(fields) >= 6:
            try:
                idx = int(fields[0])
            except ValueError:
                return
            if idx in test_records:
                # the harness reports each test exactly once; duplicates mean tampering
                test_records[idx]["passed"] = False
                test_records[idx]["error_type"] = "DuplicateResult"
                test_records[idx]["flags"] = [*test_records[idx]["flags"], "duplicate_records"]
                return
            if 0 <= idx < len(tests):
                test_records[idx] = {
                    "index": idx,
                    "passed": fields[1] == "1",
                    "error_type": fields[2] or None,
                    "message": _clip(fields[3], MAX_MESSAGE_CHARS),
                    "traceback": _clip(fields[4], MAX_TRACEBACK_CHARS),
                    "flags": [f for f in fields[5].split(",") if f],
                }
        elif kind == b"D" and phase == "tests":
            terminal = "D"
            phase = "finished"
            if fields:
                harness_flags.extend(f for f in fields[0].split(",") if f)
            finish_at = time.monotonic() + DRAIN_GRACE_SECONDS

    try:
        while open_reads:
            now = time.monotonic()
            if finish_at is not None and now >= finish_at:
                break
            if now >= deadline:
                killed_by_us = "timeout"
                break
            wait_until = deadline if finish_at is None else min(deadline, finish_at)
            events = sel.select(timeout=max(0.0, wait_until - now))
            for key, mask in events:
                fd = key.fd
                if fd == stdin_fd and mask & selectors.EVENT_WRITE:
                    if not pending:
                        try:
                            sel.unregister(stdin_fd)
                        except (KeyError, ValueError):
                            pass
                        if phase == "tests":
                            close_stdin()
                        continue
                    try:
                        n = os.write(stdin_fd, pending)
                        del pending[:n]
                    except BlockingIOError:
                        pass
                    except (BrokenPipeError, OSError):
                        close_stdin()
                    if not pending:
                        try:
                            sel.unregister(stdin_fd)
                        except (KeyError, ValueError):
                            pass
                        if phase == "tests":
                            close_stdin()
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(fd)
                    open_reads.discard(fd)
                    continue
                if fd == hp.result_r:
                    res_total += len(chunk)
                    if res_total > MAX_RESULT_CHANNEL_BYTES:
                        killed_by_us = "protocol"
                        break
                    res_buf += chunk
                    while True:
                        nl = res_buf.find(b"\n")
                        if nl < 0:
                            break
                        line = bytes(res_buf[:nl])
                        del res_buf[: nl + 1]
                        handle_record(line)
                else:
                    out_total += len(chunk)
                    buf = out[fd]
                    buf += chunk
                    if len(buf) > 2 * cap:
                        del buf[: len(buf) - cap]
                    if out_total > hard_output_cap:
                        killed_by_us = "output"
                        break
            if killed_by_us:
                break
            if hp.result_r not in open_reads and phase != "finished" and finish_at is None:
                # result pipe closed without a terminal record: drain output briefly, then stop
                finish_at = time.monotonic() + DRAIN_GRACE_SECONDS
    finally:
        sel.close()
        close_stdin()
        alive_at_kill = proc.poll() is None
        hp.kill()
        cpu_used = _reap(proc)
        for f in (proc.stdout, proc.stderr):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
        try:
            os.close(hp.result_r)
        except OSError:
            pass

    rc = proc.returncode
    if rc is not None and rc > 128 and hp.signal_via_exit_code:
        rc = -(rc - 128)  # wrappers like bwrap report "killed by signal N" as 128 + N
    result["duration_ms"] = round((time.monotonic() - start) * 1000.0, 2)
    result["stdout"] = _tail(out[next(iter(out))], cap)
    result["stderr"] = _tail(out[list(out)[1]], cap)
    result["exit_code"] = rc
    flags: list[str] = list(dict.fromkeys(harness_flags))
    if forged:
        flags.append("forged_records")
    message = result.pop("_message", "")

    # ---- assemble per-test results (default: failed / NotRun) --------------------------
    first_failure: dict[str, Any] | None = None
    for idx, rec in sorted(test_records.items()):
        tb = rec.pop("traceback")
        result["tests"][idx] = rec
        if not rec["passed"] and first_failure is None:
            first_failure = {**rec, "traceback": tb}
        for f in rec["flags"]:
            if f not in flags:
                flags.append(f)
    result["num_passed"] = sum(1 for t in result["tests"] if t["passed"])
    result["flags"] = flags
    error_types = {t["error_type"] for t in result["tests"] if not t["passed"]}

    # ---- status (precedence: syntax > timeout > memory > runtime > failed > passed) -----
    sig = -rc if rc is not None and rc < 0 else 0
    if terminal == "S":
        status = "syntax_error"
        result["tests"] = [
            {**t, "error_type": "SyntaxError", "message": "solution did not compile"}
            for t in result["tests"]
        ]
    elif (
        killed_by_us == "timeout"
        or sig == signal.SIGXCPU
        or (sig == signal.SIGKILL and not alive_at_kill and cpu_used >= limits["cpu_seconds"])
        or "Timeout" in error_types
    ):
        status = "timeout"
        result["error_type"] = "Timeout"
    elif (
        result["error_type"] == "MemoryError"
        or "MemoryError" in error_types
        or (sig == signal.SIGKILL and not alive_at_kill and terminal is None)
    ):
        status = "memory_exceeded"
        result["error_type"] = "MemoryError"
    elif terminal == "E":
        status = "runtime_error"
    elif terminal != "D":
        status = "runtime_error"
        if killed_by_us == "output":
            result["error_type"] = "OutputLimitExceeded"
        elif killed_by_us == "protocol":
            result["error_type"] = "ProtocolViolation"
        elif sig:
            result["error_type"] = f"Crashed({signal.Signals(sig).name})"
        else:
            result["error_type"] = "EarlyExit"
            result["traceback"] = (
                f"process exited (code {rc}) before all tests reported results; "
                "unreported tests count as failed"
            )
    elif result["num_passed"] == len(tests):
        status = "passed"
    else:
        status = "failed"

    if terminal == "E":
        for t in result["tests"]:
            t["error_type"] = result["error_type"]
            t["message"] = _clip("solution failed to load: " + message, MAX_MESSAGE_CHARS)
    if status in ("failed", "timeout", "memory_exceeded") and first_failure is not None:
        if not result["traceback"]:
            result["traceback"] = first_failure["traceback"]
        if result["error_type"] is None:
            result["error_type"] = first_failure["error_type"]
    if status == "timeout" and not result["traceback"]:
        result["traceback"] = (
            f"Timeout: exceeded the time limit (wall {limits['wall_seconds']}s, "
            f"cpu {limits['cpu_seconds']}s, per test {limits['per_test_seconds']}s)"
        )
    if status == "memory_exceeded" and not result["traceback"]:
        result["traceback"] = f"MemoryError: exceeded the memory limit ({limits['memory_mb']} MB)"
    result["status"] = status
    return result
