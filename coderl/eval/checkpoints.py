"""Atomic, JSON-only batch checkpoints for long benchmark runs.

These protect against a worker crash while the output filesystem survives. They cannot
preserve files after a Kaggle runtime is deleted: download the recovery archive first.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class CheckpointStore:
    def __init__(self, root: Path, protocol: dict[str, Any], *, resume: bool) -> None:
        self.root = root
        serialized = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        self.fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
        manifest = root / "manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if not resume:
                raise FileExistsError(f"{root}: checkpoints exist; use --resume or a new --out")
            if saved.get("fingerprint") != self.fingerprint:
                raise ValueError(
                    f"{root}: checkpoint protocol differs (model, data, code, decoding, "
                    "batch sizes or sandbox); use a new --out to avoid mixing evaluations"
                )
        else:
            # Never adopt orphaned batches or legacy results without their protocol manifest.
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"{root}: checkpoint manifest missing; use a new --out")
            root.mkdir(parents=True, exist_ok=True)
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"fingerprint": self.fingerprint, "protocol": protocol}, indent=2) + "\n"
            )
            temporary.replace(manifest)

    def _path(self, pass_name: str, batch: int) -> Path:
        return self.root / pass_name / f"batch-{batch:05d}.json.gz"

    def load(self, pass_name: str, batch: int, task_ids: list[str], n: int) -> dict | None:
        path = self._path(pass_name, batch)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved.get("fingerprint") != self.fingerprint:
            raise ValueError(f"{path}: checkpoint fingerprint mismatch")
        expected = {(task, index) for task in task_ids for index in range(n)}
        records = saved["records"]
        actual = [(row["task_id"], row["sample_index"]) for row in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"{path}: incomplete or duplicate checkpoint samples")
        if any(row["pass"] != pass_name for row in records):
            raise ValueError(f"{path}: checkpoint pass mismatch")
        for row in records:
            result = row["result"]
            if result["correct"] != (result["status"] == "passed" and not result["flags"]):
                raise ValueError(f"{path}: inconsistent checkpoint correctness")
        return saved

    def save(
        self,
        pass_name: str,
        batch: int,
        records: list[dict],
        generation_seconds: float,
        execution_seconds: float,
    ) -> None:
        path = self._path(pass_name, batch)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = {
            "fingerprint": self.fingerprint,
            "records": records,
            "generation_seconds": generation_seconds,
            "execution_seconds": execution_seconds,
        }
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
                compressed.write(json.dumps(payload).encode())
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
