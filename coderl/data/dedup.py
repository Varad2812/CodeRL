"""Train/eval contamination detection: exact match + MinHash-LSH near-duplicates."""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Literal

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel

from coderl.data.schema import Problem
from coderl.utils.logging import get_logger

log = get_logger(__name__)

SHINGLE_SIZE = 5
DEFAULT_THRESHOLD = 0.8
NUM_PERM = 128

_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
_DOCSTRING_QUOTES = re.compile(r"\"\"\"|'''")

Kind = Literal["exact_prompt", "exact_solution", "near_prompt", "near_solution"]


def normalize(text: str) -> str:
    """Lowercase, drop docstring quotes, punctuation and all whitespace."""
    text = _DOCSTRING_QUOTES.sub(" ", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", "", text)


def tokens(text: str) -> list[str]:
    """Lowercased word tokens (punctuation/docstring quotes act as separators)."""
    text = _DOCSTRING_QUOTES.sub(" ", text.lower()).translate(_PUNCT_TABLE)
    return text.split()


def shingles(text: str, n: int = SHINGLE_SIZE) -> set[str]:
    """Word n-gram shingles; texts shorter than ``n`` words become a single shingle."""
    toks = tokens(text)
    if not toks:
        return set()
    if len(toks) < n:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash(sh: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in sh:
        m.update(s.encode("utf-8"))
    return m


class ContaminationPair(BaseModel):
    train_id: str
    eval_id: str
    kind: Kind
    similarity: float


class ContaminationReport(BaseModel):
    n_train: int
    n_eval: int
    threshold: float
    n_exact: int
    n_near: int
    n_train_flagged: int
    flagged_train_ids: list[str]
    pairs: list[ContaminationPair]


def _field(p: Problem, which: Literal["prompt", "solution"]) -> str | None:
    return p.prompt if which == "prompt" else p.canonical_solution


def check_contamination(
    train: Sequence[Problem],
    eval_: Sequence[Problem],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = NUM_PERM,
) -> ContaminationReport:
    """Find train problems that collide with any eval problem (exact or near-duplicate)."""
    pairs: list[ContaminationPair] = []
    seen: set[tuple[str, str, str]] = set()

    def add(t: Problem, e: Problem, kind: Kind, sim: float) -> None:
        key = (t.task_id, e.task_id, kind.split("_")[1])
        if key in seen:
            return
        seen.add(key)
        pairs.append(
            ContaminationPair(train_id=t.task_id, eval_id=e.task_id, kind=kind, similarity=sim)
        )

    for which in ("prompt", "solution"):
        exact_kind: Kind = "exact_prompt" if which == "prompt" else "exact_solution"
        near_kind: Kind = "near_prompt" if which == "prompt" else "near_solution"

        # --- exact match on normalized text
        index: dict[str, list[Problem]] = {}
        eval_sh: dict[str, set[str]] = {}
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        by_id: dict[str, Problem] = {}
        for e in eval_:
            txt = _field(e, which)
            if not txt:
                continue
            index.setdefault(normalize(txt), []).append(e)
            sh = shingles(txt)
            if not sh:
                continue
            eval_sh[e.task_id] = sh
            by_id[e.task_id] = e
            lsh.insert(e.task_id, _minhash(sh, num_perm), check_duplication=False)

        for t in train:
            txt = _field(t, which)
            if not txt:
                continue
            exact_hits = {e.task_id for e in index.get(normalize(txt), [])}
            for e in index.get(normalize(txt), []):
                add(t, e, exact_kind, 1.0)
            # --- near-duplicate via MinHash LSH, verified with true Jaccard
            sh = shingles(txt)
            if not sh:
                continue
            for cand in lsh.query(_minhash(sh, num_perm)):
                if cand in exact_hits:
                    continue
                sim = jaccard(sh, eval_sh[cand])
                if sim >= threshold:
                    add(t, by_id[cand], near_kind, round(sim, 4))

    flagged = sorted({p.train_id for p in pairs})
    report = ContaminationReport(
        n_train=len(train),
        n_eval=len(eval_),
        threshold=threshold,
        n_exact=sum(p.kind.startswith("exact") for p in pairs),
        n_near=sum(p.kind.startswith("near") for p in pairs),
        n_train_flagged=len(flagged),
        flagged_train_ids=flagged,
        pairs=pairs,
    )
    log.info(
        "contamination_checked",
        n_train=len(train),
        n_eval=len(eval_),
        flagged=len(flagged),
        exact=report.n_exact,
        near=report.n_near,
    )
    return report


def remove_contaminated(train: Sequence[Problem], report: ContaminationReport) -> list[Problem]:
    """Drop flagged train problems (eval sets are never modified)."""
    flagged = set(report.flagged_train_ids)
    kept = [p for p in train if p.task_id not in flagged]
    for pid in sorted(flagged):
        log.warning("train_problem_removed_contamination", task_id=pid)
    return kept
