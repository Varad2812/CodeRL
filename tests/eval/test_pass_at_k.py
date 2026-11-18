from __future__ import annotations

from itertools import combinations
from math import comb

import pytest

from coderl.eval.pass_at_k import mean_pass_at_k, pass_at_k


def brute_force(n: int, c: int, k: int) -> float:
    """Fraction of all size-k subsets of n samples (first c correct) with >= 1 correct."""
    subsets = list(combinations(range(n), k))
    return sum(any(i < c for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_matches_brute_force(n: int) -> None:
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k), abs=1e-12)


def test_matches_closed_form_large_n() -> None:
    for n, c, k in [(20, 3, 10), (200, 17, 1), (200, 17, 100), (20, 1, 10)]:
        exact = 1 - comb(n - c, k) / comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(exact, rel=1e-12)


def test_edges() -> None:
    assert pass_at_k(20, 0, 10) == 0.0
    assert pass_at_k(20, 20, 1) == 1.0
    assert pass_at_k(20, 11, 10) == 1.0  # fewer than k incorrect -> certain
    assert pass_at_k(20, 5, 1) == pytest.approx(5 / 20)  # pass@1 == c/n


@pytest.mark.parametrize(("n", "c", "k"), [(5, 1, 10), (3, 4, 1), (3, -1, 1), (3, 1, 0)])
def test_invalid(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_mean() -> None:
    assert mean_pass_at_k([(2, 0), (2, 2)], 1) == 0.5
    with pytest.raises(ValueError):
        mean_pass_at_k([], 1)
