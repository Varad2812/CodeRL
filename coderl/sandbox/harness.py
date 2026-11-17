# coderl-sandbox-harness  (marker: used to find stray harness processes in tests)
"""Sandbox harness: runs INSIDE the sandbox, one fresh interpreter per job.

Launched by the trusted parent (``runner.py``) as::

    python -I -S -u harness.py <result_fd> <cpu_s>,<as_mb>,<fsize_mb>,<nofile>,<nproc>

STDLIB ONLY, and deliberately written in an unusual style. The model's code runs in THIS
process, so everything the harness relies on after the solution starts must be unreachable
or immutable from the solution's point of view:

* Every function/class the harness uses after ``exec(solution)`` is captured as a LOCAL of
  ``main()`` before the solution runs. Module globals are reachable through
  ``sys.modules['__main__']`` and could be rebound; locals live in frames, and every way of
  obtaining a frame (``sys._getframe``, ``tb_frame``, ``gi_frame``, ``settrace``, ``gc``
  referrers, signal handlers...) is blocked by the audit hook installed below.
* Results go to a dedicated pipe (``result_fd``), never stdout. Every record is tagged with
  a per-run nonce that only the parent and this frame know; the parent drops anything else.
* Tests (and the nonce re-check) arrive in a SECOND message, which the parent only sends
  after an authenticated "loaded" record, i.e. after the solution's top-level code is done.
  Until then the tests do not exist anywhere in this process.
* Anything unexpected -> ``os._exit`` without a "done" record. The parent treats every test
  without an authenticated result as FAILED, so failure modes can never turn into passes.

Protocol (parent -> child messages are 8-byte length-prefixed ``marshal`` blobs on fd 0)::

    M1  (nonce, solution, entry_point, protect_modules, per_test_seconds, audit_hook)
    M2  (nonce, test_setup, tests)
    records (child -> parent, one per line):  nonce \\t kind \\t hex(utf8(field)) ...
        L loaded | S syntax error | E module-level error | T test result | D done
"""

import sys


def main() -> None:  # noqa: C901 - intentionally one closure; see module docstring
    import _signal
    import _thread
    import builtins
    import gc
    import marshal
    import os
    import types  # noqa: F401 - reads tb_frame at import time; must load before the audit hook

    # ------------------------------------------------------------------ snapshots (locals!)
    _type = builtins.type
    _str = builtins.str
    _int = builtins.int
    _float = builtins.float
    _bool = builtins.bool
    _tuple = builtins.tuple
    _list = builtins.list
    _dict = builtins.dict
    _len = builtins.len
    _range = builtins.range
    _compile = builtins.compile
    _exec = builtins.exec
    _import = builtins.__import__
    _BaseException = builtins.BaseException
    _Exception = builtins.Exception
    _SyntaxError = builtins.SyntaxError
    _ValueError = builtins.ValueError
    _AttributeError = builtins.AttributeError
    _PermissionError = builtins.PermissionError
    _FunctionType = _type(main)
    _ModuleType = _type(sys)
    _type_name = _type.__dict__["__name__"].__get__  # ignores metaclass tricks
    _type_module = _type.__dict__["__module__"].__get__
    _type_mro = _type.__dict__["__mro__"].__get__
    _type_dict = _type.__dict__["__dict__"].__get__
    _module_dict = _ModuleType.__dict__["__dict__"].__get__
    _str_encode = _str.encode
    _str_split = _str.split
    _str_join = _str.join
    _str_startswith = _str.startswith
    _bytes_hex = bytes.hex
    _list_append = _list.append
    _dict_copy = _dict.copy
    _dict_get = _dict.get
    _dict_items = _dict.items
    _os_read = os.read
    _os_write = os.write
    _os_exit = os._exit
    _os_getpid = os.getpid
    _os_set_blocking = os.set_blocking
    _os_fstat = os.fstat
    _os_dup = os.dup
    _os_dup2 = os.dup2
    _os_open = os.open
    _os_close = os.close
    _marshal_loads = marshal.loads
    _exc_info = sys.exc_info
    _gc_disable = gc.disable
    _gc_enable = gc.enable
    _gc_isenabled = gc.isenabled
    _thread_count = _thread._count
    _get_ident = _thread.get_ident
    # C-level functions from _signal: the signal.py wrappers look up module globals at call
    # time, which the solution could rebind.
    _setitimer = _signal.setitimer
    _signal_c = _signal.signal
    _sigmask = _signal.pthread_sigmask
    _ITIMER_REAL = _signal.ITIMER_REAL
    _SIGALRM = _signal.SIGALRM
    _SIG_BLOCK = _signal.SIG_BLOCK
    _SIG_SETMASK = _signal.SIG_SETMASK
    _ALL_SIGNALS = _signal.valid_signals()
    _memoryview = builtins.memoryview
    _stdlib_names = sys.stdlib_module_names
    _sys_modules = sys.modules
    _builtins_dict = _module_dict(builtins)
    _pristine_builtins = _dict_copy(_builtins_dict)

    def die(code: int = 70) -> None:
        _os_exit(code)

    # ------------------------------------------------------------------ OS limits (fail closed)
    result_fd = _int(sys.argv[1])
    cpu_s, as_mb, fsize_mb, nofile, nproc = (_int(x) for x in _str_split(sys.argv[2], ","))
    strict_limits = sys.platform == "linux"
    try:
        import resource

        def _limit(res: int, value: int) -> None:
            try:
                resource.setrlimit(res, (value, value))
            except (_ValueError, OSError):
                if strict_limits:
                    raise

        _limit(resource.RLIMIT_CORE, 0)
        if cpu_s > 0:
            # soft -> SIGXCPU (terminates), hard -> SIGKILL one second later
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            except (_ValueError, OSError):
                if strict_limits:
                    raise
        if fsize_mb > 0:
            _limit(resource.RLIMIT_FSIZE, fsize_mb * 1024 * 1024)
        if nofile > 0:
            _limit(resource.RLIMIT_NOFILE, nofile)
        if strict_limits:  # RLIMIT_AS / RLIMIT_NPROC are not meaningful on macOS
            if as_mb > 0:
                _limit(resource.RLIMIT_AS, as_mb * 1024 * 1024)
            if nproc > 0:
                _limit(resource.RLIMIT_NPROC, nproc)
    except _BaseException:
        die(71)

    # Only this process may report: a fork()ed clone of the harness knows the nonce too and
    # could otherwise race a lucky guess onto the result pipe.
    harness_pid = _os_getpid()

    # Identity of the result pipe: if the solution swaps another pipe onto this fd number to
    # read our (nonce-tagged) records, the inode changes and we refuse to write.
    _st = _os_fstat(result_fd)
    result_id = (_st.st_dev, _st.st_ino)

    def read_exact(fd: int, n: int) -> bytes:
        chunks = []
        got = 0
        while got < n:
            b = _os_read(fd, n - got)
            if not b:
                die(72)
            _list_append(chunks, b)
            got += _len(b)
        return b"".join(chunks)

    def read_message(fd: int) -> object:
        n = _int.from_bytes(read_exact(fd, 8), "big")
        return _marshal_loads(read_exact(fd, n))

    m1 = read_message(0)
    if _type(m1) is not _tuple or _len(m1) != 6:
        die(73)
    nonce, solution, entry_point, protect_modules, per_test_seconds, audit_hook = m1
    if (
        _type(nonce) is not _str
        or _len(nonce) < 16
        or _type(solution) is not _str
        or _type(entry_point) is not _str
        or _type(protect_modules) is not _tuple
        or _type(per_test_seconds) is not _float
    ):
        die(73)

    # Keep a private copy of the control channel and point fd 0 at /dev/null, so input() or a
    # stray stdin read in the solution cannot consume the second message.
    control_fd = _os_dup(0)
    _null = _os_open("/dev/null", os.O_RDONLY)
    _os_dup2(_null, 0)
    _os_close(_null)

    # ------------------------------------------------------------------ result channel
    def hexf(value: object) -> str:
        if _type(value) is _int or _type(value) is _bool:
            value = _int.__repr__(_int(value))
        if _type(value) is not _str:
            value = "<non-str>"
        return _bytes_hex(_str_encode(value, "utf-8", "surrogatepass"))

    def emit(kind: str, *fields: object) -> None:
        """Write one nonce-tagged record. Critical section: no signals, no GC, no threads."""
        parts = [nonce, kind]
        for f in fields:
            _list_append(parts, hexf(f))
        data = _str_encode(_str_join("\t", parts) + "\n", "ascii")
        old_mask = _sigmask(_SIG_BLOCK, _ALL_SIGNALS)
        gc_was = _gc_isenabled()
        _gc_disable()
        try:
            if _thread_count() != 0:  # a solution thread could race the fd check below
                die(74)
            if _os_getpid() != harness_pid:
                die(78)
            st = _os_fstat(result_fd)
            if (st.st_dev, st.st_ino) != result_id:
                die(75)
            view = _memoryview(data)
            while view:
                n = _os_write(result_fd, view)
                view = view[n:]
        finally:
            if gc_was:
                _gc_enable()
            _sigmask(_SIG_SETMASK, old_mask)

    # ------------------------------------------------------------------ text helpers
    def clip(s: object, limit: int) -> str:
        if _type(s) is not _str:
            return "<non-str>"
        return s if _len(s) <= limit else s[: limit - 3] + "..."

    def clip_tail(s: str, limit: int) -> str:
        return s if _len(s) <= limit else "...\n" + s[_len(s) - limit + 4 :]

    sources: dict = {}  # filename -> list of source lines (for tracebacks)

    # ------------------------------------------------------------------ audit hook
    # "trusted" is only set while the harness itself walks a traceback, with signals blocked,
    # GC disabled and no other threads alive. Untrusted code never runs while it is set.
    trusted = [False]
    main_ident = _get_ident()
    frame_attrs = frozenset({"tb_frame", "gi_frame", "cr_frame", "ag_frame"})
    deny_events = frozenset(
        {
            # introspection that leads to harness frames / objects (tests, nonce)
            "sys._getframe",
            "sys._current_frames",
            "sys._current_exceptions",
            "sys.settrace",
            "sys.setprofile",
            "sys.addaudithook",
            "gc.get_objects",
            "gc.get_referrers",
            "gc.get_referents",
            "code.__new__",
            # processes / signals
            "os.fork",
            "os.forkpty",
            "os.exec",
            "os.spawn",
            "os.posix_spawn",
            "os.system",
            "os.startfile",
            "subprocess.Popen",
            "pty.spawn",
            "os.kill",
            "os.killpg",
            "signal.pthread_kill",
            "resource.setrlimit",
            "resource.prlimit",
            # filesystem tricks that reach /proc
            "os.chdir",
            "os.symlink",
            "os.link",
            "os.chroot",
        }
    )
    deny_prefixes = ("socket.", "ctypes.", "_ctypes.", "winreg.", "msvcrt.")
    deny_modules = frozenset(
        {
            "ctypes",
            "_ctypes",
            "_testcapi",
            "_testinternalcapi",
            "_testbuffer",
            "_testmultiphase",
            "_testimportmultiple",
            "_xxsubinterpreters",
            "_xxinterpchannels",
            "_xxtestfuzz",
            "_posixsubprocess",
            "_signal",
        }
    )

    def path_denied(p: object) -> bool:
        if _type(p) is _int:
            return False  # already-open fd
        if _type(p) is bytes:
            try:
                p = p.decode("utf-8", "surrogateescape")
            except _BaseException:
                return True
        if _type(p) is not _str:
            return True  # PathLike etc.: could return different paths on each call
        for part in _str_split(p, "/"):
            if part == "proc":
                return True
        return _str_startswith(p, ("/dev/fd", "/dev/std"))

    def hook(event: str, args: tuple) -> None:
        if _type(event) is not _str:
            raise _PermissionError("sandbox: bad audit event")
        if event == "object.__getattr__":
            if _len(args) > 1 and _type(args[1]) is _str and args[1] in frame_attrs:
                if trusted[0] and _get_ident() == main_ident:
                    return
                raise _AttributeError("sandbox: frame access is disabled")
            return
        if event == "open" or event == "os.open":
            if args and path_denied(args[0]):
                raise _PermissionError("sandbox: access to this path is disabled")
            return
        if event == "import":
            name = args[0] if args else None
            if _type(name) is _str and _str_split(name, ".")[0] in deny_modules:
                raise _PermissionError("sandbox: import of " + name + " is disabled")
            return
        if event in deny_events or _str_startswith(event, deny_prefixes):
            if event == "sys._getframe":
                # ValueError is what namedtuple/typing/enum/logging already tolerate
                raise _ValueError("sandbox: frame access is disabled")
            raise _PermissionError("sandbox: " + event + " is disabled")

    # ------------------------------------------------------------------ tracebacks
    def frames_of(tb: object) -> list:
        """Walk a traceback in trusted mode, collecting (filename, lineno, name) only."""
        out: list = []
        old_mask = _sigmask(_SIG_BLOCK, _ALL_SIGNALS)
        gc_was = _gc_isenabled()
        _gc_disable()
        try:
            if _thread_count() != 0:
                return out
            trusted[0] = True
            try:
                n = 0
                while tb is not None and n < 4000:
                    code = tb.tb_frame.f_code
                    _list_append(out, (code.co_filename, tb.tb_lineno, code.co_name))
                    tb = tb.tb_next
                    n += 1
            finally:
                trusted[0] = False
        except _BaseException:
            pass
        finally:
            if gc_was:
                _gc_enable()
            _sigmask(_SIG_SETMASK, old_mask)
        return out

    def format_tb(frames: list, header: str) -> tuple:
        """Keep only solution/test frames. Returns (text, last solution line number)."""
        lines = ["Traceback (most recent call last):"]
        last_solution_line = 0
        kept = []
        for fn, ln, name in frames:
            if _type(fn) is _str and (fn == "<solution>" or _str_startswith(fn, "<test")):
                _list_append(kept, (fn, ln, name))
        if _len(kept) > 30:
            kept = kept[:5] + [None] + kept[-24:]
        for item in kept:
            if item is None:
                _list_append(lines, "  [... frames omitted ...]")
                continue
            fn, ln, name = item
            if fn == "<solution>" and _type(ln) is _int:
                last_solution_line = ln
            name = name if _type(name) is _str else "?"
            _list_append(lines, '  File "' + fn + '", line ' + num(ln) + ", in " + name)
            src = _dict_get(sources, fn)
            if src is not None and _type(ln) is _int and 0 < ln <= _len(src):
                _list_append(lines, "    " + src[ln - 1].strip())
        _list_append(lines, header)
        return clip_tail(_str_join("\n", lines), 2000), last_solution_line

    def num(n: object) -> str:
        return _int.__repr__(n) if _type(n) is _int else "?"

    def exc_name(t: type) -> str:
        """Like Python's traceback: `ValueError`, `re.error`, `MyError` (solution-defined)."""
        name = _type_name(t)
        if _type(name) is not _str:
            return "<exception>"
        if t is TestTimeout:
            return name
        try:
            mod = _type_module(t)
        except _BaseException:
            return name
        if _type(mod) is not _str or mod in ("builtins", "__solution__", "__main__"):
            return name
        return mod + "." + name

    def describe(ev: object) -> tuple:
        """(type name, message) of an exception. str(ev) is untrusted code -> time-boxed."""
        tname = exc_name(_type(ev))
        msg = ""
        armed[0] = True
        _setitimer(_ITIMER_REAL, 0.5)
        try:
            msg = _str(ev)
        except _BaseException:
            msg = "<unprintable exception>"
        finally:
            armed[0] = False
            _setitimer(_ITIMER_REAL, 0)
        return tname, clip(msg, 500)

    # ------------------------------------------------------------------ per-test timer
    class TestTimeout(_BaseException):
        pass

    TestTimeout.__name__ = "Timeout"
    TestTimeout.__qualname__ = "Timeout"
    armed = [False]

    def on_alarm(signum: int, frame: object) -> None:
        if armed[0]:
            armed[0] = False
            raise TestTimeout("test exceeded its time budget")

    # ------------------------------------------------------------------ eq-hack detection
    simple_types = frozenset(
        {_int, _float, _bool, _str, bytes, complex, _type(None), _list, _tuple, _dict, set}
    )
    simple_types = simple_types | {frozenset}
    suspicious_dunders = ("__eq__", "__ne__", "__bool__")

    def custom_dunder(value: object) -> bool:
        t = _type(value)
        if t in simple_types:
            return False
        for cls in _type_mro(t):
            d = _type_dict(cls)
            for name in suspicious_dunders:
                if name in d:
                    mod = d.get("__module__", "builtins")
                    top = _str_split(mod, ".")[0] if _type(mod) is _str else ""
                    if top not in _stdlib_names:
                        return True
        return False

    def suspicious(value: object) -> bool:
        try:
            if custom_dunder(value):
                return True
            t = _type(value)
            if t is _list or t is _tuple:
                for item in value[:100]:  # type: ignore[index]
                    if custom_dunder(item):
                        return True
        except _BaseException:
            return True
        return False

    def probe(fn: object, flags: list) -> object:
        def candidate(*args: object, **kwargs: object) -> object:
            result = fn(*args, **kwargs)  # type: ignore[operator]
            if suspicious(result) and "custom_eq" not in flags:
                _list_append(flags, "custom_eq")
            return result

        return candidate

    # ------------------------------------------------------------------ compile the solution
    sources["<solution>"] = _str_split(solution, "\n")
    try:
        code = _compile(solution, "<solution>", "exec", dont_inherit=True, optimize=0)
    except _BaseException:
        _, ev, _ = _exc_info()
        if _type(ev) is _SyntaxError or isinstance(ev, _SyntaxError):
            lineno = ev.lineno if _type(ev.lineno) is _int else 0  # type: ignore[union-attr]
            offset = ev.offset if _type(ev.offset) is _int else 0  # type: ignore[union-attr]
            text = ev.text if _type(ev.text) is _str else ""  # type: ignore[union-attr]
            msg = ev.msg if _type(ev.msg) is _str else "invalid syntax"  # type: ignore[union-attr]
            tname = _type_name(_type(ev))
            caret = " " * (max(offset - 1, 0) + 4) + "^" if offset else ""
            tb = '  File "<solution>", line ' + num(lineno) + "\n    " + text.strip()
            tb = tb + ("\n" + caret if caret else "") + "\n" + tname + ": " + msg
            emit("S", tname, clip(msg, 500), clip_tail(tb, 2000), lineno)
        else:
            tname = _type_name(_type(ev))
            emit("S", tname, "could not compile solution", tname, 0)
        die(0)

    # ------------------------------------------------------------------ lock down, then exec
    protected: list = []  # (module, snapshot of its __dict__)
    for mod_name in protect_modules:
        try:
            mod = _import(mod_name)
            _list_append(protected, (mod_name, mod, _dict_copy(_module_dict(mod))))
        except _BaseException:
            pass

    def blocked_signal(*args: object, **kwargs: object) -> object:
        raise _PermissionError("sandbox: installing signal handlers is disabled")

    _signal_c(_SIGALRM, on_alarm)
    if audit_hook:
        # Python signal handlers receive the interrupted frame (a way back to our frames), so
        # the solution may not install any. signal.py re-exports _signal.signal at import time
        # and calls it through the module at runtime, so patching _signal covers both; a fresh
        # re-import of _signal is denied by the hook.
        _module_dict(_signal)["signal"] = blocked_signal
        sys.addaudithook(hook)

    sol_mod = _ModuleType("__solution__")
    sol_ns = _module_dict(sol_mod)
    sol_ns["__builtins__"] = _dict_copy(_pristine_builtins)
    _sys_modules["__solution__"] = sol_mod

    try:
        _exec(code, sol_ns)
    except _BaseException:
        _, ev, tb = _exc_info()
        frames = frames_of(tb)
        tname, msg = describe(ev)
        text, lineno = format_tb(frames, tname + (": " + msg if msg else ""))
        emit("E", tname, msg, text, lineno)
        die(0)
    del code

    emit("L")

    # ------------------------------------------------------------------ receive the tests
    try:
        _os_set_blocking(control_fd, True)  # the solution may have flipped it
    except _BaseException:
        die(76)
    m2 = read_message(control_fd)
    if _type(m2) is not _tuple or _len(m2) != 3 or _type(m2[0]) is not _str or m2[0] != nonce:
        die(76)  # not from our parent
    _, test_setup, tests = m2
    if _type(test_setup) is not _str or _type(tests) is not _tuple:
        die(76)

    # ------------------------------------------------------------------ build test namespace
    # Test code sees the solution's names (entry point, helper classes like Node) but pristine
    # builtins, so `abs = lambda x: 0` or `builtins.abs = ...` cannot change how asserts work.
    test_flags: list = []
    base: dict = {}
    try:
        for k, v in _dict_items(sol_ns):
            if _type(k) is _str and (k not in _pristine_builtins or k == entry_point):
                base[k] = v
    except _BaseException:
        die(77)
    for mod_name, mod, _snap in protected:
        current = _dict_get(base, mod_name)
        if current is None or _type(current) is _ModuleType:
            base[mod_name] = mod
    target = _dict_get(sol_ns, entry_point)
    if _type(target) is _FunctionType:
        base[entry_point] = probe(target, test_flags)
    global_flags: list = []

    def restore() -> None:
        """Undo monkeypatching of builtins and of modules the tests use."""
        changed = False
        for k, v in _dict_items(_pristine_builtins):
            if _dict_get(_builtins_dict, k, base) is not v:
                _builtins_dict[k] = v
                changed = True
        if changed and "builtins_modified" not in global_flags:
            _list_append(global_flags, "builtins_modified")
        for _name, mod, snap in protected:
            md = _module_dict(mod)
            mchanged = False
            for k, v in _dict_items(snap):
                if _dict_get(md, k, base) is not v:
                    md[k] = v
                    mchanged = True
            if mchanged and "modules_modified" not in global_flags:
                _list_append(global_flags, "modules_modified")

    def run_code(src: str, filename: str, ns: dict) -> object:
        """Run src in ns under the per-test budget. Returns None or (type, msg, tb, line)."""
        sources[filename] = _str_split(src, "\n")
        try:
            restore()
            c = _compile(src, filename, "exec", dont_inherit=True, optimize=0)
            ns["__builtins__"] = _dict_copy(_pristine_builtins)
            _signal_c(_SIGALRM, on_alarm)
            armed[0] = True
            _setitimer(_ITIMER_REAL, per_test_seconds)
            try:
                _exec(c, ns)
            finally:
                armed[0] = False
                _setitimer(_ITIMER_REAL, 0)
            return None
        except _BaseException:
            armed[0] = False
            _setitimer(_ITIMER_REAL, 0)
            _, ev, tb = _exc_info()
            frames = frames_of(tb)
            tname, msg = describe(ev)
            text, lineno = format_tb(frames, tname + (": " + msg if msg else ""))
            return (tname, msg, text, lineno)

    setup_error = None
    if test_setup:
        setup_error = run_code(test_setup, "<test setup>", base)

    for i in _range(_len(tests)):
        src = tests[i]
        del test_flags[:]
        if setup_error is not None:
            tname, msg, text, _ln = setup_error  # type: ignore[misc]
            emit("T", i, 0, tname, "test setup failed: " + msg, text, "")
            continue
        if _type(src) is not _str:
            emit("T", i, 0, "SandboxError", "invalid test", "", "")
            continue
        outcome = run_code(src, "<test " + num(i) + ">", _dict_copy(base))
        flags = _str_join(",", test_flags)
        if outcome is None:
            emit("T", i, 1, "", "", "", flags)
        else:
            tname, msg, text, _ln = outcome  # type: ignore[misc]
            emit("T", i, 0, tname, msg, text, flags)

    restore()
    emit("D", _str_join(",", global_flags))
    die(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 - any harness failure = no "done" record = all fail
        import os

        os._exit(70)
