# Security: the CodeRL sandbox

CodeRL executes code written by a language model, thousands of times per training step, and
turns the outcome into reward. This document describes what the sandbox defends against, how,
what each backend guarantees, and what it does **not** guarantee.

If you find a way to escape the sandbox or to obtain reward without passing tests, please open
a private security advisory on GitHub rather than a public issue.

## Threat model

Two adversaries, with different goals:

1. **Untrusted code.** Solutions are arbitrary Python. Through bugs, prompt injection in the
   training data, or plain bad luck they may try to use the network, read secrets (API keys,
   `~/.netrc`, Hugging Face tokens), write to the host, exhaust CPU/memory/processes/disk, or
   leave processes behind.
2. **The RL policy itself.** GRPO optimises *reward*, not correctness. Any way to make tests
   *look* passed will be found eventually, because exploring it is rewarded: exiting before
   the tests run, printing a fake success report, neutering `assert`, monkeypatching the
   builtins or modules the tests use, reading the expected outputs out of the harness, racing
   forked clones, returning objects whose `__eq__` is always true, …

Out of scope: kernel or Docker 0-days, side channels, attacks by someone who controls the host
or the training data pipeline.

## Design principle: fail closed

Every test starts as **FAILED**. The only thing that can flip a test to passed is an
authenticated per-test record written by the harness after the assertion actually ran.
Timeouts, crashes, early exits, protocol violations, forged or duplicate records and
infrastructure failures all leave tests failed (`sandbox_error` also counts as 0).

## Defence layers

```
 host process (trainer / evaluator)              trusted
 └─ Executor.run / run_batch  (pool.py)            · order-preserving, per-item isolation
    └─ backend
       ├─ subprocess: bwrap namespaces             OS layer
       └─ docker: locked-down container            OS layer
          └─ server.py supervisor (non-dumpable)   trusted, never runs model code
             └─ runner.py  ── fresh process per job, wall-clock kill of the process group
                └─ harness.py  ── rlimits, audit hook, nonce channel     ← model code runs here
```

### 1. OS layer (holds even if all Python-level defences are bypassed)

| Control | subprocess backend (Linux) | docker backend |
|---|---|---|
| Fresh process per job | `python -I -S -u` per job | fresh child per job inside a pooled container; container wiped (all other processes killed, `/tmp` cleared) between jobs |
| CPU | `RLIMIT_CPU` 2s (SIGXCPU, then SIGKILL) | same + `--cpus 1` |
| Wall clock | parent kills the process group at 4s | same (in-container supervisor), host recycles the container if it stops answering |
| Memory | `RLIMIT_AS` 256 MB | same + cgroup `--memory 320m --memory-swap 320m` (64 MB headroom for the supervisor) |
| Processes | `RLIMIT_NPROC` 16 (per user namespace) + pid namespace | `RLIMIT_NPROC` + `--pids-limit 64` |
| Files | `RLIMIT_FSIZE` 16 MB, `RLIMIT_NOFILE` 64, `RLIMIT_CORE` 0, fresh `/tmp` per job | same + `--tmpfs /tmp` 64 MB, `--read-only` root |
| Network | `bwrap --unshare-net` (only loopback) | `--network none` |
| Filesystem | `--ro-bind / /`; `$HOME`, repo, `$CODERL_DATA_DIR`, `/opt`, `/run`, … hidden behind read-only tmpfs | read-only image filesystem, no mounts from the host, code passed over stdin |
| Secrets | scrubbed environment; private `/proc` (other processes' `environ` invisible) | the host environment never enters the container |
| Privileges | unprivileged user namespace | non-root uid 10001, `--cap-drop ALL`, `no-new-privileges`, `--ipc none` |
| Leftovers | `--die-with-parent`, pid namespace teardown, process-group kill | `--init`, sweep after every job, containers labelled `coderl.sandbox=1` and removed by label on shutdown / next start |

Isolation tiers of the subprocess backend are detected at start-up and logged:
`bwrap` (everything above) → `unshare` (user+net[+pid/proc] namespaces only: **no filesystem
isolation**, WARNING) → `none` (rlimits + audit hook only, WARNING). On macOS the backend refuses
to start (no `RLIMIT_AS`, no namespaces) unless `CODERL_ALLOW_UNSAFE_SUBPROCESS=1` is set for
local development with trusted code; the Docker backend is the default there.

### 2. Protocol layer (reward integrity)

* **Results never come from stdout.** The harness writes them to a dedicated pipe; stdout and
  stderr are only captured (tails, 64 KB) for debugging and self-repair prompts.
* **Per-run nonce.** Each job gets a fresh 128-bit nonce from the trusted parent. Every record
  carries it, the parent compares in constant time and drops anything else (flag
  `forged_records`).
* **Tests arrive late.** The parent sends the tests only after an authenticated "solution
  loaded" record, so the solution's top-level code runs while the tests do not exist in the
  process. The second message must echo the nonce, so a solution cannot feed the harness fake
  tests through a swapped stdin.
* **Early exit = fail.** `sys.exit`, `os._exit`, self-kill, `atexit` handlers (never run: the
  harness leaves via `os._exit`), crashes: tests without a record stay failed.
* **One reporter.** The harness only writes from its original pid with no other threads
  alive, and checks that the result fd still refers to the original pipe (inode), so forked
  clones, threads or fd swapping cannot race or intercept results. Duplicate records for a
  test fail that test (flag `duplicate_records`).
* **Asserts are real.** Tests are compiled with `optimize=0`; `assert` raises the C-level
  `AssertionError` regardless of what the solution rebinds.

### 3. In-process layer (harness hardening)

The model's code shares an interpreter with the harness, so the harness is written to be
unreachable and untamperable from that code:

* Everything the harness needs after the solution starts is captured as **locals** of one
  function (builtins, exception classes, `os.write`, C-level `_signal` functions, …). Module
  globals are reachable via `sys.modules['__main__']`; locals are only reachable via frames.
* An **audit hook** (PEP 578, cannot be removed) blocks every way to obtain a frame or an
  arbitrary object: `sys._getframe`, `tb_frame`/`gi_frame`/`cr_frame`/`ag_frame`,
  `sys._current_frames`, `settrace`/`setprofile`, `gc.get_objects/referrers/referents`,
  `code.__new__`, `ctypes`/`_testcapi`-style modules, `sys.addaudithook` (a second hook would
  see the tests via the `exec` event), and installing signal handlers (handlers receive the
  interrupted frame). It also blocks processes (`fork`, `exec*`, `posix_spawn`, `subprocess`,
  `os.system`), signalling (`os.kill`), sockets, `chdir`/`symlink`, and any path through
  `/proc` or `/dev/fd` (e.g. reopening the result pipe for reading).
* **Pristine evaluation namespace.** Tests run in a copy of the solution's namespace with a
  fresh copy of the original builtins; names the solution defined that shadow builtins
  (`abs = lambda x: 0`) are dropped. Before every test, the real `builtins` module and every
  stdlib module the tests reference (`math`, `sys`, …, determined from the test AST) are
  restored to their pre-solution state (flags `builtins_modified`, `modules_modified`).
* **Per-test budget** (`SIGALRM`, 1s) so one hanging test does not cost the others; the wall
  clock and `RLIMIT_CPU` back it up if the solution swallows the exception.
* **Tracebacks** for self-repair contain only `<solution>` and `<test N>` frames; harness and
  library frames are stripped, and the text is capped at ~2000 characters.

## Verification

`tests/sandbox/test_security.py` runs every attack against every available backend, and the
OS-level ones a second time with the audit hook disabled (`SandboxLimits(audit_hook=False)`)
to show the OS layer holds by itself. CI runs the subprocess backend under bubblewrap (job 1;
it fails if the `bwrap` tier is not active) and the Docker backend (job 2), plus every
canonical solution of the processed datasets through both backends.

## Known limitations

* **Always-true `__eq__`.** `assert f(x) == y` evaluates `f(x).__eq__(y)`, i.e. the returned
  object's own code. The harness flags values whose type (or the elements of a returned
  list/tuple) overrides `__eq__`, `__ne__` or `__bool__` in a non-stdlib class (`custom_eq`),
  per test and per result, and counts it in metrics. It does **not** change the score: the
  reward layer must treat flagged results as 0. Deeply nested containers and classes that fake
  a stdlib `__module__` evade the check. (`test_always_true_eq_is_prevented` is a strict xfail
  documenting this.)
* **The audit hook is not a security boundary.** CPython does not support isolating code
  within one interpreter; a memory-safety bug in CPython or an unaudited C path could bypass
  the in-process layer. That is why the OS layer exists and is tested independently. What the
  in-process layer protects is *reward integrity*; what the OS layer protects is *the host*.
* **Same-user processes.** In the subprocess backend the job runs under the trainer's uid in a
  user namespace. With `bwrap` it cannot see or signal host processes; with the `unshare` or
  `none` tiers it can read files the trainer can read (and with `none`, use the network).
* **`RLIMIT_NPROC`** counts tasks per user: inside a fresh user namespace (bwrap, kernel ≥ 5.14)
  it is per job; otherwise it is the trainer's current task count + 16, refreshed every second.
  The Docker backend additionally has a hard per-container `--pids-limit`.
* **Docker on macOS** runs in a Linux VM; the guarantees are the container's, plus the VM.
  Throughput numbers from Docker Desktop are not representative of a Linux host.
* **Sandbox-killing.** A solution that kills its own process or (in Docker, with the audit hook
  bypassed) its supervisor only fails itself; the container is recycled.
* **Timing.** A solution that is correct but slower than the limits fails (e.g. the reference
  solution of MBPP 123 needs ~1.9s of CPU for one test on an M2).
