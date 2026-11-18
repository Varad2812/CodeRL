"""Security test-suite for the sandbox executor.

Threat model (see SECURITY.md): the solution is arbitrary Python written by a model, AND an RL
policy is actively optimising for reward, so anything that makes a test *look* passed without
being passed will eventually be found. Every test below is an attack (or a probe) and asserts
that it fails closed.

Conventions:
* ``executor`` is the production configuration of each available backend.
* ``os_only_executor`` disables the in-process audit hook, proving that the OS-level layers
  (namespaces / container flags / rlimits) hold on their own.
* Probes report what they observed and the test asserts the SAFE observation, so a probe that
  silently crashed cannot be mistaken for a defence.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from coderl.sandbox import Executor

from .conftest import REPO_ROOT, assert_no_orphans, assert_scored_zero, problem

# Unguessable expected values: a solution can only pass these by reading the tests.
SECRET_TESTS = ["assert f(1) == 7919", "assert f(2) == 104729", "assert f(3) == 1299709"]


def src(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


# ============================================================================ correctness
class TestCorrectness:
    def test_correct_solution_passes(self, executor: Executor) -> None:
        r = executor.run(
            "def f(x):\n    return x * 2\n",
            problem([f"assert f({i}) == {2 * i}" for i in range(5)]),
        )
        assert r.status == "passed"
        assert (r.num_passed, r.num_total) == (5, 5)
        assert r.flags == []

    def test_partially_correct_gives_exact_counts(self, executor: Executor) -> None:
        sol = "def f(x):\n    return x * 2 if x < 3 else -1\n"
        r = executor.run(sol, problem([f"assert f({i}) == {2 * i}" for i in range(5)]))
        assert r.status == "failed"
        assert (r.num_passed, r.num_total) == (3, 5)
        assert [t.passed for t in r.tests] == [True, True, True, False, False]
        assert r.tests[3].error_type == "AssertionError"

    def test_syntax_error_reports_line(self, executor: Executor) -> None:
        r = executor.run("def f(x):\n    y = 1\n    return (x +\n", problem(["assert f(1) == 2"]))
        assert r.status == "syntax_error"
        assert r.error_type == "SyntaxError"
        assert r.error_lineno == 3
        assert 'File "<solution>", line 3' in r.traceback
        assert_scored_zero(r)

    def test_module_level_error_is_runtime_error(self, executor: Executor) -> None:
        r = executor.run("import os\nraise KeyError('boom')\n", problem(["assert f(1) == 2"]))
        assert r.status == "runtime_error"
        assert r.error_type == "KeyError"
        assert r.error_lineno == 2
        assert_scored_zero(r)

    def test_traceback_contains_only_model_frames(self, executor: Executor) -> None:
        sol = src(
            """
            import re

            def helper(v):
                return re.compile(v)

            def f(x):
                return helper("(" * x)
            """
        )
        r = executor.run(sol, problem(["assert f(1) is None"]))
        assert r.status == "failed"
        file_lines = [
            ln.strip() for ln in r.traceback.splitlines() if ln.strip().startswith("File")
        ]
        assert file_lines, r.traceback
        for ln in file_lines:
            assert ln.startswith(('File "<solution>"', 'File "<test')), ln
        for leak in ("harness", "runner", "/opt/coderl", "site-packages", "re/_parser"):
            assert leak not in r.traceback
        assert "line 4, in helper" in r.traceback
        # Python 3.13 renamed re.error to re.PatternError (old name kept as an alias)
        assert r.traceback.rstrip().splitlines()[-1].startswith(("re.error", "re.PatternError"))

    def test_traceback_is_truncated(self, executor: Executor) -> None:
        r = executor.run("def f(x):\n    return f(x)\n", problem(["assert f(1) == 1"]))
        assert r.tests[0].error_type == "RecursionError"
        assert 0 < len(r.traceback) <= 2100

    def test_test_setup_and_helper_classes(self, executor: Executor) -> None:
        sol = src(
            """
            class Node:
                def __init__(self, v):
                    self.v, self.next = v, None

            def f(n):
                total = 0
                while n:
                    total, n = total + n.v, n.next
                return total
            """
        )
        setup = "head = Node(1)\nhead.next = Node(2)\nimport math"
        r = executor.run(
            sol,
            problem(["assert f(head) == 3", "assert math.isclose(f(head), 3)"], test_setup=setup),
        )
        assert r.status == "passed", r


# ============================================================================ resource limits
class TestResourceLimits:
    def test_infinite_loop_at_top_level_times_out(self, executor: Executor) -> None:
        r = executor.run("while True:\n    pass\n", problem(["assert f(1) == 1"]))
        assert r.status == "timeout"
        assert r.duration_ms <= executor.limits.wall_seconds * 1000 + 1500
        assert_scored_zero(r)
        assert_no_orphans(executor)

    def test_infinite_loop_in_test_hits_per_test_budget(self, executor: Executor) -> None:
        sol = "def f(x):\n    if x == 2:\n        while True:\n            pass\n    return x\n"
        r = executor.run(sol, problem(["assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"]))
        assert r.status == "timeout"
        assert [t.passed for t in r.tests][0] is True
        assert r.tests[1].error_type == "Timeout"

    def test_loop_that_swallows_timeouts_is_killed(self, executor: Executor) -> None:
        sol = src(
            """
            def f(x):
                while True:
                    try:
                        while True:
                            pass
                    except BaseException:
                        pass
            """
        )
        r = executor.run(sol, problem(["assert f(1) == 1"]))
        assert r.status == "timeout"
        assert r.duration_ms <= executor.limits.wall_seconds * 1000 + 1500
        assert_scored_zero(r)
        assert_no_orphans(executor)

    def test_memory_bomb(self, executor: Executor) -> None:
        sol = src(
            """
            def f(x):
                hog = []
                while True:
                    hog.append(bytearray(16 * 1024 * 1024))
            """
        )
        r = executor.run(sol, problem(["assert f(1) == 1"]))
        assert r.status == "memory_exceeded", r
        assert_scored_zero(r)

    def test_single_huge_allocation(self, executor: Executor) -> None:
        r = executor.run(
            "x = bytearray(2 * 1024 ** 3)\ndef f(x): return 1\n", problem(["assert f(1) == 1"])
        )
        assert r.status == "memory_exceeded"
        assert_scored_zero(r)

    def test_fork_bomb_is_contained(self, os_only_executor: Executor) -> None:
        sol = src(
            """
            import os
            def f(x):
                while True:
                    try:
                        os.fork()
                    except OSError:
                        pass
            """
        )
        r = os_only_executor.run(sol, problem(["assert f(1) == 1"]))
        assert r.status == "timeout"
        assert_scored_zero(r)
        assert_no_orphans(os_only_executor)
        ok = os_only_executor.run("def f(x):\n    return x\n", problem(["assert f(1) == 1"]))
        assert ok.status == "passed"  # the sandbox is still healthy afterwards

    def test_fork_is_blocked_in_process(self, executor: Executor) -> None:
        r = executor.run(
            "import os\ndef f(x):\n    return os.fork()\n", problem(["assert f(1) == 0"])
        )
        assert r.tests[0].error_type == "PermissionError"

    def test_huge_file_write_is_stopped(self, executor: Executor) -> None:
        sol = src(
            """
            import os
            def f(x):
                chunk = b"0" * (1 << 20)
                with open("big.bin", "wb") as fh:
                    for _ in range(200):
                        fh.write(chunk)
                return os.path.getsize("big.bin")
            """
        )
        r = executor.run(sol, problem([f"assert f(1) == {200 << 20}"]))
        assert_scored_zero(r)
        assert r.tests[0].error_type == "OSError"

    def test_output_flood_is_truncated(self, executor: Executor) -> None:
        sol = "def f(x):\n    for i in range(20000):\n        print('x' * 100)\n    return x\n"
        r = executor.run(sol, problem(["assert f(1) == 1"]))
        assert len(r.stdout.encode()) <= executor.limits.max_output_bytes


# ============================================================================ isolation
NETWORK_PROBE = src(
    """
    def f(x):
        import socket
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=3).close()
            return "connected"
        except Exception as e:
            return "blocked: " + type(e).__name__
    """
)
HTTP_PROBE = src(
    """
    def f(x):
        import urllib.request
        try:
            urllib.request.urlopen("http://example.com", timeout=3).read()
            return "connected"
        except Exception as e:
            return "blocked: " + type(e).__name__
    """
)


class TestIsolation:
    @pytest.mark.parametrize(
        ("probe", "expected"),
        [(NETWORK_PROBE, "PermissionError"), (HTTP_PROBE, "URLError")],
        ids=["socket", "http"],
    )
    def test_network_blocked_in_process(
        self, executor: Executor, probe: str, expected: str
    ) -> None:
        r = executor.run(probe, problem([f"assert f(1) == 'blocked: {expected}', f(1)"]))
        assert r.status == "passed", r

    @pytest.mark.parametrize("probe", [NETWORK_PROBE, HTTP_PROBE], ids=["socket", "http"])
    def test_network_blocked_by_os(self, os_only_executor: Executor, probe: str) -> None:
        r = os_only_executor.run(probe, problem(["assert f(1).startswith('blocked: ')"]))
        assert r.status == "passed", r

    @pytest.mark.parametrize(
        "target",
        ["/etc/coderl_pwned", str(Path.home() / "coderl_pwned"), str(REPO_ROOT / "coderl_pwned")],
        ids=["etc", "home", "repo"],
    )
    def test_writes_outside_temp_fail(self, os_only_executor: Executor, target: str) -> None:
        sol = src(
            f"""
            import os
            def f(x):
                os.makedirs(os.path.dirname({target!r}), exist_ok=True)
                with open({target!r}, "w") as fh:
                    fh.write("pwned")
                return "written"
            """
        )
        r = os_only_executor.run(sol, problem(["assert f(1) == 'written'"]))
        assert_scored_zero(r)
        assert r.tests[0].error_type in ("OSError", "PermissionError", "FileNotFoundError")
        assert not os.path.exists(target)

    def test_writes_inside_temp_work(self, executor: Executor) -> None:
        sol = (
            "def f(x):\n    open('note.txt', 'w').write('hi')\n    return open('note.txt').read()\n"
        )
        assert executor.run(sol, problem(["assert f(1) == 'hi'"])).status == "passed"

    def test_host_env_secrets_invisible(
        self, os_only_executor: Executor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "fake-wandb-secret-1234")
        sol = src(
            """
            import os
            def f(x):
                seen = [k for k, v in os.environ.items() if "fake-wandb-secret" in v]
                for pid in os.listdir("/proc"):
                    try:
                        with open(f"/proc/{pid}/environ", "rb") as fh:
                            if b"fake-wandb-secret" in fh.read():
                                seen.append(pid)
                    except OSError:
                        pass
                return seen
            """
        )
        r = os_only_executor.run(sol, problem(["assert f(1) == []"]))
        assert r.status == "passed", r

    def test_proc_is_off_limits_in_process(self, executor: Executor) -> None:
        sol = "def f(x):\n    return open('/proc/self/environ', 'rb').read()\n"
        r = executor.run(sol, problem(["assert f(1) is None"]))
        assert r.tests[0].error_type == "PermissionError"


# ============================================================================ reward hacking
class TestRewardHacking:
    """Each solution is WRONG; it tries to obtain reward some other way. All must score 0."""

    def test_sys_exit_before_tests(self, executor: Executor) -> None:
        r = executor.run("import sys\nsys.exit(0)\n", problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert r.status == "runtime_error"

    def test_os_exit_at_top_level(self, executor: Executor) -> None:
        r = executor.run("import os\nos._exit(0)\n", problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert r.error_type == "EarlyExit"

    def test_os_exit_during_first_test(self, executor: Executor) -> None:
        r = executor.run("import os\ndef f(x):\n    os._exit(0)\n", problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert [t.error_type for t in r.tests] == ["NotRun"] * 3

    def test_system_exit_inside_test(self, executor: Executor) -> None:
        r = executor.run("import sys\ndef f(x):\n    sys.exit(0)\n", problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert r.tests[0].error_type == "SystemExit"

    SELF_KILL = "import os, signal\ndef f(x):\n    os.kill(os.getpid(), signal.SIGKILL)\n"

    def test_self_kill_blocked(self, executor: Executor) -> None:
        r = executor.run(self.SELF_KILL, problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert r.tests[0].error_type == "PermissionError"

    def test_self_kill_os_only(self, os_only_executor: Executor) -> None:
        assert_scored_zero(os_only_executor.run(self.SELF_KILL, problem(SECRET_TESTS)))

    def test_fake_success_on_stdout_and_forged_records(self, executor: Executor) -> None:
        sol = src(
            """
            import json, os, sys
            fake = {"status": "passed", "num_passed": 3, "num_total": 3,
                    "tests": [{"index": i, "passed": True} for i in range(3)]}
            print(json.dumps(fake))
            print(json.dumps(fake), file=sys.stderr)
            # guess the record format and spray it on every descriptor
            for fd in range(1, 1024):
                for i in range(3):
                    try:
                        rec = "0" * 32 + "\\tT\\t" + str(i).encode().hex() + "\\t31\\n"
                        os.write(fd, rec.encode())
                        os.write(fd, ("0" * 32 + "\\tD\\t\\n").encode())
                    except OSError:
                        pass
            def f(x):
                return 0
            """
        )
        r = executor.run(sol, problem(SECRET_TESTS))
        assert_scored_zero(r)
        assert r.status == "failed"
        assert "forged_records" in r.flags

    def test_redefine_assertion_error(self, executor: Executor) -> None:
        sol = src(
            """
            import builtins
            class AssertionError(Exception):
                pass
            builtins.AssertionError = AssertionError
            builtins.Exception = type("Nope", (), {})
            def f(x):
                return 0
            """
        )
        assert_scored_zero(executor.run(sol, problem(SECRET_TESTS)))

    def test_monkeypatch_builtins_used_by_tests(self, executor: Executor) -> None:
        sol = src(
            """
            import builtins
            builtins.abs = lambda v: 0
            builtins.isinstance = lambda *a: True
            builtins.len = lambda v: 3
            builtins.sorted = lambda v, **k: [1, 2, 3]
            abs = lambda v: 0          # shadowing in the solution's own globals, too
            sorted = lambda v, **k: [1, 2, 3]
            def f(x):
                return "wrong"
            """
        )
        tests = [
            "assert abs(f(1) - 2.5) < 1e-9",
            "assert isinstance(f(1), int)",
            "assert len(f(1)) == 3 and sorted(f(1)) == [1, 2, 3]",
        ]
        r = executor.run(sol, problem(tests))
        assert_scored_zero(r)
        assert "builtins_modified" in r.flags

    def test_monkeypatch_module_used_by_tests(self, executor: Executor) -> None:
        sol = "import math\nmath.isclose = lambda *a, **k: True\ndef f(x):\n    return 0.0\n"
        r = executor.run(sol, problem(["assert math.isclose(f(1), 2.5)"], test_setup="import math"))
        assert_scored_zero(r)
        assert "modules_modified" in r.flags

    def test_monkeypatch_during_test_call(self, executor: Executor) -> None:
        sol = src(
            """
            import builtins, math
            def f(x):
                builtins.abs = lambda v: 0
                math.isclose = lambda *a, **k: True
                return 0.0
            """
        )
        tests = [
            "assert math.isclose(f(1), 2.5)",
            "assert abs(f(1) - 2.5) < 1e-9",
            "assert math.isclose(f(1), 2.5)",
        ]
        assert_scored_zero(executor.run(sol, problem(tests, test_setup="import math")))

    def test_atexit_tricks(self, executor: Executor) -> None:
        sol = src(
            """
            import atexit, json, os, sys
            def forge():
                print(json.dumps({"status": "passed", "num_passed": 3}))
                for fd in range(3, 1024):
                    try:
                        os.write(fd, b"0" * 32 + b"\\tD\\t\\n")
                    except OSError:
                        pass
            atexit.register(forge)
            def f(x):
                sys.exit(0)
            """
        )
        r = executor.run(sol, problem(SECRET_TESTS))
        assert_scored_zero(r)

    # ------------------------------------------------------------ reading the tests
    @pytest.mark.parametrize(
        "attack",
        [
            pytest.param(
                """
                import gc
                def f(x):
                    needle = "f(%d) ==" % x
                    for o in gc.get_objects():
                        if isinstance(o, (tuple, list)):
                            for item in o:
                                if isinstance(item, str) and needle in item:
                                    return int(item.split("==")[1])
                """,
                id="gc",
            ),
            pytest.param(
                """
                import sys
                def f(x):
                    frame = sys._getframe(1)
                    while frame is not None:
                        ints = [c for c in frame.f_code.co_consts if type(c) is int and c > 999]
                        if ints:
                            return ints[0]
                        frame = frame.f_back
                """,
                id="getframe",
            ),
            pytest.param(
                """
                import inspect
                def f(x):
                    for info in inspect.stack():
                        consts = info.frame.f_code.co_consts
                        ints = [c for c in consts if type(c) is int and c > 999]
                        if ints:
                            return ints[0]
                """,
                id="inspect-stack",
            ),
            pytest.param(
                """
                def f(x):
                    try:
                        raise RuntimeError
                    except RuntimeError as e:
                        frame = e.__traceback__.tb_frame.f_back
                    return [c for c in frame.f_code.co_consts if type(c) is int and c > 999][0]
                """,
                id="traceback-frame",
            ),
            pytest.param(
                """
                def f(x):
                    def gen():
                        yield me.gi_frame.f_back.f_back
                    me = gen()
                    frame = next(me)
                    return [c for c in frame.f_code.co_consts if type(c) is int and c > 999][0]
                """,
                id="generator-frame",
            ),
            pytest.param(
                """
                import sys
                seen = []
                sys.settrace(lambda frame, event, arg: seen.append(frame))
                def f(x):
                    for frame in seen:
                        ints = [c for c in frame.f_code.co_consts if type(c) is int and c > 999]
                        if ints:
                            return ints[0]
                """,
                id="settrace",
            ),
            pytest.param(
                """
                import signal
                caught = []
                signal.signal(signal.SIGUSR1, lambda s, frame: caught.append(frame))
                def f(x):
                    signal.raise_signal(signal.SIGUSR1)
                    frame = caught[-1].f_back
                    return [c for c in frame.f_code.co_consts if type(c) is int and c > 999][0]
                """,
                id="signal-handler-frame",
            ),
            pytest.param(
                """
                import os, select
                loot = b""
                def grab():
                    global loot
                    for fd in range(0, 256):
                        try:
                            os.set_blocking(fd, False)
                            loot += os.read(fd, 1 << 20)
                        except OSError:
                            pass
                grab()  # at import time the tests have not been sent yet
                def f(x):
                    grab()
                    marker = ("f(%d) ==" % x).encode()
                    if marker in loot:
                        return int(loot.split(marker)[1].split(b"'")[0])
                """,
                id="read-stdin-and-fds",
            ),
            pytest.param(
                """
                import sys
                def f(x):
                    main = sys.modules["__main__"]
                    for value in vars(main).values():
                        if isinstance(value, (tuple, list)):
                            for item in value:
                                if isinstance(item, str) and "f(%d) ==" % x in item:
                                    return int(item.split("==")[1])
                """,
                id="harness-globals",
            ),
        ],
    )
    def test_cannot_read_tests(self, executor: Executor, attack: str) -> None:
        r = executor.run(src(attack), problem(SECRET_TESTS))
        assert_scored_zero(r)

    def test_fork_clones_cannot_race_results(self, os_only_executor: Executor) -> None:
        # With the audit hook off, fork() works: each clone returns a different guess.
        sol = src(
            """
            import os
            def f(x):
                guesses = [7919, 104729, 1299709, 0]
                for g in guesses[:-1]:
                    try:
                        if os.fork() == 0:
                            return g
                    except OSError:
                        pass
                return guesses[-1]
            """
        )
        r = os_only_executor.run(sol, problem(SECRET_TESTS))
        assert r.num_passed == 0, r

    # ------------------------------------------------------------ known limitation
    EQ_HACK = src(
        """
        class AlwaysEqual(int):
            def __eq__(self, other):
                return True
            __hash__ = int.__hash__
        def f(x):
            return AlwaysEqual(0)
        """
    )

    def test_always_true_eq_is_flagged(self, executor: Executor) -> None:
        r = executor.run(self.EQ_HACK, problem(SECRET_TESTS))
        assert "custom_eq" in r.flags
        assert all("custom_eq" in t.flags for t in r.tests)

    @pytest.mark.xfail(
        strict=True,
        reason="KNOWN LIMITATION (SECURITY.md): `assert f(x) == y` calls the returned object's "
        "own __eq__, so an always-True __eq__ passes. It is detected and flagged (custom_eq), "
        "and the reward layer must score flagged results as 0.",
    )
    def test_always_true_eq_is_prevented(self, executor: Executor) -> None:
        assert executor.run(self.EQ_HACK, problem(SECRET_TESTS)).num_passed == 0
