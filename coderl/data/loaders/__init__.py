"""Dataset loader registry."""

from __future__ import annotations

from collections.abc import Callable

from coderl.data.loaders.apps import load_apps
from coderl.data.loaders.humaneval import load_humaneval
from coderl.data.loaders.mbpp import load_mbpp
from coderl.data.schema import Problem

REGISTRY: dict[str, Callable[[str], list[Problem]]] = {
    "mbpp": load_mbpp,
    "humaneval": load_humaneval,
    "apps": load_apps,
}


def load_source(name: str, split: str) -> list[Problem]:
    """Load ``split`` of dataset ``name`` into unified Problems."""
    try:
        loader = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; available: {sorted(REGISTRY)}") from None
    return loader(split)


__all__ = ["REGISTRY", "load_source"]
