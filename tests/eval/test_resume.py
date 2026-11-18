"""A generation failure must not discard an earlier completed sandbox batch."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from coderl.data.schema import write_problems
from coderl.eval.run_benchmarks import EvalConfig, run_eval
from coderl.gen.mock import MockGenerator, canonical_responder

from .test_run_benchmarks import HE, FakeExecutor


def config(tmp_path: Path, **kwargs: object) -> EvalConfig:
    data = tmp_path / "data"
    write_problems(data / "eval_humaneval.jsonl", HE)
    return EvalConfig(
        model="mock/model",
        out=tmp_path / "out",
        data_dir=data,
        benchmarks=["humaneval"],
        batch_size=1,
        n_samples=20,
        **kwargs,
    )


@pytest.mark.parametrize("wrong_every", [0, 4])
def test_resume_after_next_batch_oom_preserves_all_samples(
    tmp_path: Path, wrong_every: int
) -> None:
    class FailingGenerator(MockGenerator):
        def generate(self, *args: object, **kwargs: object) -> list:
            # Fail while the prior sandbox batch may still be running. The worker
            # must persist its result even though the main thread never collects it.
            if self.calls:
                raise RuntimeError("simulated CUDA out of memory")
            return super().generate(*args, **kwargs)

    cfg = config(tmp_path)
    responder = canonical_responder(HE, wrong_every=wrong_every)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_eval(FailingGenerator(responder), cfg, FakeExecutor())
    root = cfg.out / "humaneval"
    assert (root / "checkpoints/greedy/batch-00000.json.gz").is_file()
    assert not (root / "summary.json").exists(), "partial runs must not publish final metrics"

    cfg.resume = True
    generator = MockGenerator(responder)
    (summary,) = run_eval(generator, cfg, FakeExecutor())
    # The first greedy problem is reused, the remaining greedy and sampled batches run.
    assert len(generator.calls) == 3
    assert generator.calls[0]["seed"] == cfg.seed + 1
    assert summary["passes"]["greedy"]["resumed_batches"] == 1
    assert summary["metrics"] == {
        "pass@1_greedy": 0.0 if wrong_every else 1.0,
        "pass@1_sampled": 0.75 if wrong_every else 1.0,
        "pass@10_sampled": 1.0,
    }
    with gzip.open(root / "samples.jsonl.gz", "rt") as stream:
        samples = [json.loads(line) for line in stream]
    keys = {(r["pass"], r["task_id"], r["sample_index"]) for r in samples}
    assert len(keys) == len(samples) == 42

    # A further resume reuses every batch and preserves the original pass@k results.
    complete = MockGenerator(responder)
    (again,) = run_eval(complete, cfg, FakeExecutor())
    assert complete.calls == []
    assert again["metrics"] == summary["metrics"]


@pytest.mark.parametrize("changed", ["seed", "n_samples", "max_new_tokens", "batch_size"])
def test_resume_rejects_changed_protocol_before_overwriting(tmp_path: Path, changed: str) -> None:
    cfg = config(tmp_path)
    run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
    output = cfg.out / "humaneval/summary.json"
    original = output.read_bytes()
    cfg.resume = True
    setattr(cfg, changed, getattr(cfg, changed) + 1)
    with pytest.raises(ValueError, match="protocol differs"):
        run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
    assert output.read_bytes() == original


def test_rerun_requires_explicit_resume(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
    with pytest.raises(FileExistsError, match="--resume"):
        run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())


def test_resume_rejects_changed_tests(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
    modified = HE[0].model_copy(update={"tests": ["assert False"]})
    write_problems(cfg.data_dir / "eval_humaneval.jsonl", [modified, HE[1]])
    cfg.resume = True
    with pytest.raises(ValueError, match="protocol differs"):
        run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())


def test_resume_rejects_duplicate_saved_samples(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
    batch = cfg.out / "humaneval/checkpoints/sampled/batch-00000.json.gz"
    with gzip.open(batch, "rt") as stream:
        saved = json.load(stream)
    saved["records"][1] = saved["records"][0]
    with gzip.open(batch, "wt") as stream:
        json.dump(saved, stream)
    cfg.resume = True
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        run_eval(MockGenerator(canonical_responder(HE)), cfg, FakeExecutor())
