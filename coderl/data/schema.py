"""Unified problem schema shared by all data sources."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Problem(BaseModel):
    task_id: str  # unique, prefixed e.g. "mbpp/11"
    source: Literal["mbpp", "humaneval", "apps"]
    split: Literal["train", "validation", "test", "prompt"]
    prompt: str
    entry_point: str
    tests: list[str]  # each item is ONE independently runnable assert
    test_setup: str = ""
    canonical_solution: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def write_problems(path: str | Path, problems: Iterable[Problem]) -> int:
    """Write problems as JSONL; returns number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for prob in problems:
            f.write(prob.model_dump_json() + "\n")
            n += 1
    return n


def read_problems(path: str | Path) -> list[Problem]:
    with Path(path).open(encoding="utf-8") as f:
        return [Problem.model_validate_json(line) for line in f if line.strip()]
