"""Unbiased pass@k estimator (Chen et al., 2021, "Evaluating Large Language Models Trained on
Code", Fig. 3): ``1 - C(n-c, k) / C(n, k)``, computed as a numerically stable product."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of ``k`` samples (drawn without replacement from ``n``
    samples of which ``c`` are correct) is correct."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < k:
        raise ValueError(f"need n >= k samples, got n={n}, k={k}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(counts: Iterable[tuple[int, int]], k: int) -> float:
    """Mean pass@k over problems given ``(n, c)`` per problem."""
    vals = [pass_at_k(n, c, k) for n, c in counts]
    if not vals:
        raise ValueError("no problems")
    return float(np.mean(vals))
