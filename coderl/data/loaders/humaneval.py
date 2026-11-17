"""HumanEval loader (EVAL ONLY) -> unified Problem schema.

The ``check(candidate)`` function is split into individually runnable asserts so that
per-test pass fractions can be computed; on any failure we fall back to a single test.
"""

from __future__ import annotations

import ast
from typing import Any

from coderl.data.schema import Problem
from coderl.utils.logging import get_logger

log = get_logger(__name__)

HF_PATH = "openai/openai_humaneval"


def _load_hf(split: str) -> Any:
    """Fetch HumanEval from the HF Hub. Monkeypatched in tests."""
    from datasets import load_dataset

    return load_dataset(HF_PATH, split=split)


class _Rename(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node


def _has_assert(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Assert) for n in ast.walk(node))


def _src(node: ast.AST, old: str, new: str) -> str:
    import copy

    renamed = _Rename(old, new).visit(copy.deepcopy(node))
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed)


def split_check_function(test_src: str, entry_point: str) -> tuple[list[str], str]:
    """Split HumanEval ``test`` source into (tests, test_setup).

    Raises ValueError if the source cannot be split into at least one test.
    """
    try:
        tree = ast.parse(test_src)
    except SyntaxError as exc:
        raise ValueError(f"unparseable test source: {exc}") from exc

    check_fn: ast.FunctionDef | None = None
    setup_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
        else:
            setup_nodes.append(node)
    if check_fn is None or not check_fn.args.args:
        raise ValueError("no check(candidate) function found")
    param = check_fn.args.args[0].arg

    setup: list[str] = [ast.unparse(n) for n in setup_nodes]
    context: list[str] = []  # non-assert statements that later asserts may depend on
    tests: list[str] = []
    for stmt in check_fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring / bare constant
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            setup.append(_src(stmt, param, entry_point))
        elif _has_assert(stmt):
            # a bare assert, or a compound block (loop/if/with) containing asserts
            tests.append("\n".join([*context, _src(stmt, param, entry_point)]))
        else:
            context.append(_src(stmt, param, entry_point))
    if not tests:
        raise ValueError("check() contains no asserts")
    return tests, "\n".join(setup)


def _row_to_problem(row: dict[str, Any], split: str) -> Problem:
    entry_point = row["entry_point"]
    metadata: dict[str, Any] = {}
    try:
        tests, test_setup = split_check_function(row["test"], entry_point)
        metadata["test_split"] = "split"
    except ValueError as exc:
        log.warning("humaneval_split_fallback", task_id=row["task_id"], reason=str(exc))
        test_src = row["test"].strip()
        tests = [f"{test_src}\n\ncheck({entry_point})"]
        test_setup = ""
        metadata["test_split"] = "fallback"
        metadata["fallback_reason"] = str(exc)
    metadata["canonical_solution_is_full_program"] = True
    task_num = row["task_id"].split("/")[-1]
    return Problem(
        task_id=f"humaneval/{task_num}",
        source="humaneval",
        split=split,  # type: ignore[arg-type]
        prompt=row["prompt"],
        entry_point=entry_point,
        tests=tests,
        test_setup=test_setup,
        # full runnable program: prompt (signature/docstring/helpers) + body
        canonical_solution=row["prompt"] + row["canonical_solution"],
        metadata=metadata,
    )


def load_humaneval(split: str = "test") -> list[Problem]:
    if split != "test":
        raise ValueError("HumanEval only has a 'test' split (EVAL ONLY)")
    problems = [_row_to_problem(dict(r), split) for r in _load_hf(split)]
    n_split = sum(p.metadata["test_split"] == "split" for p in problems)
    log.info(
        "humaneval_loaded", n=len(problems), split_ok=n_split, fallback=len(problems) - n_split
    )
    return problems
