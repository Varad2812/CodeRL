"""MBPP loader (full config) -> unified Problem schema."""

from __future__ import annotations

import ast
import builtins
from typing import Any

from coderl.data.schema import Problem
from coderl.utils.logging import get_logger

log = get_logger(__name__)

HF_PATH = "google-research-datasets/mbpp"
HF_CONFIG = "full"
SPLITS = ("train", "validation", "test", "prompt")


def _load_hf(split: str) -> Any:
    """Fetch an MBPP split from the HF Hub. Monkeypatched in tests."""
    from datasets import load_dataset

    return load_dataset(HF_PATH, HF_CONFIG, split=split)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _called_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    return [n for n in names if not hasattr(builtins, n)]


def _defined_functions(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]


def parse_entry_point(code: str, first_test: str) -> str:
    """Pick the entry point: the function the first test calls, via ``ast``."""
    defined = _defined_functions(code)
    called = _called_names(first_test)
    for name in called:
        if name in defined:
            return name
    if called:  # solution defines it oddly (e.g. assigned lambda / class) — trust the test
        return called[0]
    if defined:
        return defined[-1]
    raise ValueError("could not determine entry point")


def _row_to_problem(row: dict[str, Any], split: str) -> Problem:
    code = _normalize_newlines(row["code"])
    tests = [_normalize_newlines(t).strip() for t in row["test_list"]]
    challenge = [_normalize_newlines(t).strip() for t in (row.get("challenge_test_list") or [])]
    if not tests:
        raise ValueError(f"mbpp/{row['task_id']} has no tests")
    entry_point = parse_entry_point(code, tests[0])
    # Prompt shows the description AND first assert so the model learns name/signature.
    prompt = f"{row['text'].strip()}\nYour code should pass this test:\n{tests[0]}"
    return Problem(
        task_id=f"mbpp/{row['task_id']}",
        source="mbpp",
        split=split,  # type: ignore[arg-type]
        prompt=prompt,
        entry_point=entry_point,
        tests=tests + challenge,
        test_setup=_normalize_newlines(row.get("test_setup_code") or "").strip(),
        canonical_solution=code,
        metadata={
            "text": row["text"].strip(),
            "n_challenge_tests": len(challenge),
            "challenge_test_indices": list(range(len(tests), len(tests) + len(challenge))),
            "first_test_in_prompt": True,
        },
    )


def load_mbpp(split: str) -> list[Problem]:
    if split not in SPLITS:
        raise ValueError(f"unknown MBPP split {split!r}; expected one of {SPLITS}")
    problems: list[Problem] = []
    for row in _load_hf(split):
        try:
            problems.append(_row_to_problem(dict(row), split))
        except ValueError as exc:
            log.warning("mbpp_row_skipped", task_id=row.get("task_id"), error=str(exc))
    log.info("mbpp_loaded", split=split, n=len(problems))
    return problems
