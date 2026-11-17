"""APPS loader — stub."""

from __future__ import annotations

from coderl.data.schema import Problem


def load_apps(split: str) -> list[Problem]:
    # TODO: APPS problems are stdin/stdout based, not assert based. Supporting them needs a
    # separate test format (input/expected-output pairs) in the schema and sandbox. Maybe later.
    raise NotImplementedError("APPS is stdin-based and not supported yet")
