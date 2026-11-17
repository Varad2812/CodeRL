"""Smoke test: fixture canonical solutions must pass their own formatted tests.

Uses plain exec() on trusted, hand-written fixtures only; the real sandbox comes later.
"""

import pytest

from coderl.data.loaders import load_source
from coderl.data.schema import Problem


def _all(fake: None) -> list[Problem]:
    return [
        *load_source("mbpp", "train"),
        *load_source("mbpp", "validation"),
        *load_source("mbpp", "test"),
        *load_source("humaneval", "test"),
    ]


def test_canonical_solutions_pass_tests(fake_datasets: None) -> None:
    probs = _all(None)
    assert probs
    for p in probs:
        assert p.canonical_solution
        for t in p.tests:
            ns: dict[str, object] = {}
            exec(p.test_setup + "\n" + p.canonical_solution, ns)
            exec(p.test_setup + "\n" + t, ns)  # each test runs independently


def test_broken_solution_fails_tests(fake_datasets: None) -> None:
    p = load_source("humaneval", "test")[0]
    ns: dict[str, object] = {}
    exec("def add(a, b):\n    return a - b", ns)
    with pytest.raises(AssertionError):
        exec(p.test_setup + "\n" + p.tests[0], ns)
