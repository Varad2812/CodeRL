"""Extract a runnable program from a model completion.

1. Pick the code: the first fenced block tagged python (preferring one that defines the entry
   point) -> ``code_block``; else any fenced block -> ``fallback``; else the raw text with
   leading/trailing prose trimmed -> ``raw``; nothing left -> ``empty``.
2. Strip test scaffolding the model added despite instructions: ``if __name__ == "__main__":``
   blocks, top-level asserts, bare top-level calls (``print(...)``, ``main()``) and top-level
   statements that call the entry point (``result = f(3)``).
3. HumanEval: if only a function body came back, re-attach the prompt's signature + docstring;
   prepend imports / helper functions from the prompt that the completion does not define
   (the official protocol executes prompt + completion, so this context is always available).
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from coderl.data.schema import Problem

ExtractionStatus = Literal["code_block", "fallback", "raw", "empty"]

PYTHON_TAGS = {"python", "py", "python3", "py3"}
_FENCE = re.compile(
    r"^[ \t]*```[ \t]*(?P<lang>[\w+.-]*)[^\n]*\n(?P<body>.*?)(?:^[ \t]*```|\Z)",
    re.DOTALL | re.MULTILINE,
)
_CODE_START = re.compile(r"^(def |async def |class |import |from \S+ import |@)")
_MAX_TRIM_LINES = 40


@dataclass(frozen=True)
class Extraction:
    code: str
    status: ExtractionStatus
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- step 1: find code
def _blocks(text: str) -> list[tuple[str, str]]:
    return [
        (m.group("lang").lower(), textwrap.dedent(m.group("body"))) for m in _FENCE.finditer(text)
    ]


def _defines(code: str, name: str) -> bool:
    return (
        re.search(rf"^\s*(async\s+)?def\s+{re.escape(name)}\s*\(", code, re.MULTILINE) is not None
    )


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def _trim_prose(text: str) -> str:
    """Raw output: start at the first code-looking line, drop trailing prose until it parses."""
    lines = text.strip("\n").splitlines()
    start = next((i for i, ln in enumerate(lines) if _CODE_START.match(ln)), None)
    if start is None:
        return text.strip("\n")
    lines = lines[start:]
    for cut in range(min(_MAX_TRIM_LINES, len(lines) - 1) + 1):
        candidate = "\n".join(lines[: len(lines) - cut] if cut else lines)
        if _parses(candidate):
            return candidate
    return "\n".join(lines)


def select_code(text: str, entry_point: str) -> tuple[str, ExtractionStatus]:
    blocks = _blocks(text)
    python = [b for lang, b in blocks if lang in PYTHON_TAGS and b.strip()]
    if python:
        return next((b for b in python if _defines(b, entry_point)), python[0]), "code_block"
    other = [b for _, b in blocks if b.strip()]
    if other:
        return next((b for b in other if _defines(b, entry_point)), other[0]), "fallback"
    if not text.strip() or (blocks and not other):
        return "", "empty"
    return _trim_prose(text), "raw"


# ---------------------------------------------------------------------- step 2: strip tests
def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    parts = [node.test.left, *node.test.comparators]
    names = {p.id for p in parts if isinstance(p, ast.Name)}
    consts = {p.value for p in parts if isinstance(p, ast.Constant)}
    return "__name__" in names and "__main__" in consts


def _calls(node: ast.AST, names: set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name) and fn.id in names:
                return True
    return False


def strip_test_code(code: str, entry_point: str) -> tuple[str, list[str]]:
    """Remove top-level test scaffolding. Unparseable code is cleaned line-by-line."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return _strip_lines(code)
    lines = code.splitlines(keepends=True)
    drop: set[int] = set()
    notes: list[str] = []
    for node in tree.body:
        reason = None
        if _is_main_guard(node):
            reason = "stripped_main_guard"
        elif isinstance(node, ast.Assert):
            reason = "stripped_asserts"
        elif isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            reason = "stripped_top_level_calls"
        elif not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
        ) and _calls(node, {entry_point, "print"}):
            reason = "stripped_top_level_calls"
        if reason:
            end = node.end_lineno or node.lineno
            drop.update(range(node.lineno - 1, end))
            if reason not in notes:
                notes.append(reason)
    if not drop:
        return code, []
    kept = "".join(ln for i, ln in enumerate(lines) if i not in drop)
    return kept.rstrip() + "\n", notes


def _strip_lines(code: str) -> tuple[str, list[str]]:
    out: list[str] = []
    notes: list[str] = []
    in_guard = False
    for ln in code.splitlines():
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln):
            in_guard = True
            notes.append("stripped_main_guard")
            continue
        if in_guard:
            if not ln.strip() or ln[:1] in (" ", "\t"):
                continue
            in_guard = False
        if re.match(r"^(print\(|assert\s)", ln):
            notes.append(
                "stripped_top_level_calls" if ln.startswith("print") else "stripped_asserts"
            )
            continue
        out.append(ln)
    return "\n".join(out).rstrip() + "\n", list(dict.fromkeys(notes))


# ---------------------------------------------------------------------- step 3: humaneval
def _looks_like_body(code: str, entry_point: str) -> bool:
    if _defines(code, entry_point):
        return False
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    if first[:1] in (" ", "\t"):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, ast.Return) for n in tree.body)


def _reattach_signature(body: str, prompt: str) -> str:
    first = next((ln for ln in body.splitlines() if ln.strip()), "")
    if first[:1] not in (" ", "\t"):
        body = textwrap.indent(body, "    ")
    return prompt.rstrip("\n") + "\n" + body.strip("\n") + "\n"


def _prompt_context(prompt: str, code: str, entry_point: str) -> tuple[str, list[str]]:
    """Imports and helper defs from the prompt that the completion lacks."""
    try:
        tree = ast.parse(prompt)
    except (SyntaxError, ValueError):
        return "", []
    pieces: list[str] = []
    notes: list[str] = []
    for node in tree.body:
        src = ast.get_source_segment(prompt, node)
        if not src:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if src.strip() not in code:
                pieces.append(src)
                if "added_prompt_imports" not in notes:
                    notes.append("added_prompt_imports")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name != entry_point and not re.search(
                rf"^\s*(async\s+)?(def|class)\s+{re.escape(node.name)}\b", code, re.MULTILINE
            ):
                pieces.append(src)
                if "added_prompt_helpers" not in notes:
                    notes.append("added_prompt_helpers")
    return ("\n\n".join(pieces) + "\n\n\n") if pieces else "", notes


# ---------------------------------------------------------------------- public API
def extract_code(text: str, problem: Problem) -> Extraction:
    code, status = select_code(text, problem.entry_point)
    if status == "empty" or not code.strip():
        return Extraction(code="", status="empty")
    notes: list[str] = []
    if problem.source == "humaneval" and _looks_like_body(code, problem.entry_point):
        code = _reattach_signature(code, problem.prompt)
        notes.append("reattached_signature")
    code, stripped = strip_test_code(code, problem.entry_point)
    notes += stripped
    if problem.source == "humaneval" and "reattached_signature" not in notes:
        context, ctx_notes = _prompt_context(problem.prompt, code, problem.entry_point)
        code = context + code
        notes += ctx_notes
    if not code.strip():
        return Extraction(code="", status="empty", notes=notes)
    return Extraction(code=code, status=status, notes=notes)
