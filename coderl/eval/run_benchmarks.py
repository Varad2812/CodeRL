"""Measure pass@k on HumanEval / MBPP-test.

    python -m coderl.eval.run_benchmarks --model Qwen/Qwen2.5-Coder-0.5B-Instruct \\
        --backend vllm --benchmarks humaneval,mbpp_test --out results/base/qwen0.5b

Per benchmark, two passes: GREEDY (T=0, n=1) -> pass@1 greedy, and SAMPLED (T=0.8,
top_p=0.95, n=20) -> unbiased pass@1 / pass@10. Every completion is executed in the sandbox
with the ``eval`` limit profile. A completion counts as correct only if the sandbox reports
``passed`` AND raised no integrity flags. Writes ``<out>/<benchmark>/summary.json`` (small,
committed) and ``samples.jsonl.gz`` (every completion; gitignored).
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any

import typer

from coderl.data.schema import Problem, read_problems
from coderl.eval.extract import Extraction, extract_code
from coderl.eval.pass_at_k import mean_pass_at_k
from coderl.eval.prompts import SYSTEM_PROMPT, render_prompt
from coderl.gen.base import Completion, Generator
from coderl.sandbox import ExecutionResult, Executor, get_executor
from coderl.utils.logging import get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Evaluate a model on HumanEval / MBPP-test.")

BENCHMARKS = {"humaneval": "eval_humaneval.jsonl", "mbpp_test": "eval_mbpp_test.jsonl"}
STATUSES = (
    "passed",
    "failed",
    "syntax_error",
    "runtime_error",
    "timeout",
    "memory_exceeded",
    "sandbox_error",
)
SECRET_ENV = ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
PACKAGES = ("coderl", "torch", "transformers", "vllm", "accelerate", "peft", "numpy", "pydantic")


@dataclass(frozen=True)
class PassSpec:
    name: str  # "greedy" | "sampled"
    n: int
    temperature: float
    top_p: float
    ks: tuple[int, ...]


@dataclass
class EvalConfig:
    model: str
    out: Path
    benchmarks: list[str] = field(default_factory=lambda: list(BENCHMARKS))
    adapter: str | None = None
    limit: int | None = None
    data_dir: Path = Path("data/processed")
    n_samples: int = 20
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    seed: int = 1234
    batch_size: int = 32  # problems per generation/execution batch
    greedy: bool = True
    sampled: bool = True
    sandbox_workers: int | None = None

    def passes(self) -> list[PassSpec]:
        out = []
        if self.greedy:
            out.append(PassSpec("greedy", 1, 0.0, 1.0, (1,)))
        if self.sampled:
            ks = tuple(k for k in (1, 10) if k <= self.n_samples)
            out.append(PassSpec("sampled", self.n_samples, self.temperature, self.top_p, ks))
        return out


def is_correct(r: ExecutionResult) -> bool:
    return r.status == "passed" and not r.flags


def _result_summary(r: ExecutionResult) -> dict[str, Any]:
    return {
        "status": r.status,
        "correct": is_correct(r),
        "num_passed": r.num_passed,
        "num_total": r.num_total,
        "error_type": r.error_type,
        "error_lineno": r.error_lineno,
        "flags": r.flags,
        "duration_ms": r.duration_ms,
        "traceback": r.traceback[-600:],
    }


# ---------------------------------------------------------------------- environment metadata
def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    env: dict[str, Any] = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "packages": versions,
    }
    torch = sys.modules.get("torch")  # only report GPUs if a backend already loaded torch
    if torch is not None:
        try:
            if torch.cuda.is_available():
                env["gpus"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                env["cuda"] = torch.version.cuda
        except Exception:  # noqa: BLE001 - metadata must never fail a run
            pass
    return env


def sandbox_info(executor: Executor) -> dict[str, Any]:
    isolation = getattr(executor, "isolation", None) or (
        "docker" if executor.name == "docker" else "unknown"
    )
    return {
        "backend": executor.name,
        "isolation": isolation,
        "profile": "eval",
        "limits": executor.limits.model_dump(),
    }


# ---------------------------------------------------------------------- one pass
def _diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    statuses = Counter(r["result"]["status"] for r in records)
    flags = Counter(f for r in records for f in r["result"]["flags"])
    return {
        "num_completions": n,
        "status_counts": {s: statuses.get(s, 0) for s in STATUSES},
        "syntax_error_rate": statuses.get("syntax_error", 0) / n if n else 0.0,
        "timeout_rate": statuses.get("timeout", 0) / n if n else 0.0,
        "runtime_error_rate": statuses.get("runtime_error", 0) / n if n else 0.0,
        "truncated_rate": sum(r["finish_reason"] == "length" for r in records) / n if n else 0.0,
        "extraction_status": dict(Counter(r["extraction_status"] for r in records)),
        "extraction_notes": dict(Counter(x for r in records for x in r["extraction_notes"])),
        "avg_completion_tokens": sum(r["num_tokens"] for r in records) / n if n else 0.0,
        "flagged_results": sum(1 for r in records if r["result"]["flags"]),
        "flags": dict(flags),
    }


def run_pass(
    gen: Generator,
    executor: Executor,
    problems: Sequence[Problem],
    spec: PassSpec,
    cfg: EvalConfig,
    benchmark: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate + execute one pass. Execution of batch i overlaps generation of batch i+1."""
    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], ExecutionResult] = {}
    correct: dict[str, int] = {p.task_id: 0 for p in problems}
    prompts = [render_prompt(p, gen) for p in problems]
    n_batches = (len(problems) + cfg.batch_size - 1) // cfg.batch_size
    t_gen = t_exec = 0.0
    start = time.monotonic()

    def execute(
        batch: Sequence[Problem], comps: list[list[Completion]], extracted: list[list[Extraction]]
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.monotonic()
        todo: dict[tuple[str, str], tuple[str, Problem]] = {}
        for p, exs in zip(batch, extracted, strict=True):
            for ex in exs:
                key = (p.task_id, ex.code)
                if key not in cache and key not in todo:
                    todo[key] = (ex.code, p)
        results = executor.run_batch(list(todo.values()), max_workers=cfg.sandbox_workers)
        cache.update(zip(todo, results, strict=True))
        out = []
        for p, cs, exs in zip(batch, comps, extracted, strict=True):
            for j, (c, ex) in enumerate(zip(cs, exs, strict=True)):
                r = cache[(p.task_id, ex.code)]
                out.append(
                    {
                        "task_id": p.task_id,
                        "benchmark": benchmark,
                        "pass": spec.name,
                        "sample_index": j,
                        "completion": c.text,
                        "num_tokens": c.num_tokens,
                        "finish_reason": c.finish_reason,
                        "extraction_status": ex.status,
                        "extraction_notes": ex.notes,
                        "code": ex.code,
                        "result": _result_summary(r),
                    }
                )
        return out, time.monotonic() - t0

    def collect(fut: Future[tuple[list[dict[str, Any]], float]], batch_idx: int) -> None:
        nonlocal t_exec
        recs, secs = fut.result()
        t_exec += secs
        records.extend(recs)
        for r in recs:
            correct[r["task_id"]] += int(r["result"]["correct"])
        done = {r["task_id"] for r in records}
        log.info(
            "eval_progress",
            benchmark=benchmark,
            pass_name=spec.name,
            batch=batch_idx + 1,
            batches=n_batches,
            problems_done=len(done),
            problems=len(problems),
            running_pass_rate=round(sum(r["result"]["correct"] for r in records) / len(records), 4),
            gen_seconds=round(t_gen, 1),
            exec_seconds=round(t_exec, 1),
            elapsed_seconds=round(time.monotonic() - start, 1),
        )

    pending: tuple[Future[tuple[list[dict[str, Any]], float]], int] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-exec") as pool:
        for b in range(n_batches):
            lo, hi = b * cfg.batch_size, min((b + 1) * cfg.batch_size, len(problems))
            batch = problems[lo:hi]
            t0 = time.monotonic()
            comps = gen.generate(
                prompts[lo:hi],
                n=spec.n,
                temperature=spec.temperature,
                top_p=spec.top_p,
                max_new_tokens=cfg.max_new_tokens,
                seed=cfg.seed + lo,  # per-problem seed = seed + problem index
            )
            t_gen += time.monotonic() - t0
            extracted = [
                [extract_code(c.text, p) for c in cs] for p, cs in zip(batch, comps, strict=True)
            ]
            if pending is not None:
                collect(*pending)
            pending = (pool.submit(execute, batch, comps, extracted), b)
        if pending is not None:
            collect(*pending)

    counts = [(spec.n, correct[p.task_id]) for p in problems]
    metrics = {f"pass@{k}": mean_pass_at_k(counts, k) for k in spec.ks}
    stats = {
        "n": spec.n,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "metrics": metrics,
        "problems_solved": sum(1 for _, c in counts if c > 0),
        "unique_programs_executed": len(cache),
        "generation_seconds": round(t_gen, 2),
        "execution_seconds": round(t_exec, 2),
        "wall_seconds": round(time.monotonic() - start, 2),
        "diagnostics": _diagnostics(records),
    }
    return records, stats


# ---------------------------------------------------------------------- one benchmark
def load_benchmark(name: str, data_dir: Path, limit: int | None) -> list[Problem]:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; expected one of {list(BENCHMARKS)}")
    path = data_dir / BENCHMARKS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m coderl.data.build` first")
    problems = read_problems(path)
    return problems[:limit] if limit else problems


def evaluate(gen: Generator, executor: Executor, cfg: EvalConfig, benchmark: str) -> dict[str, Any]:
    problems = load_benchmark(benchmark, cfg.data_dir, cfg.limit)
    out_dir = cfg.out / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    passes: dict[str, Any] = {}
    samples_path = out_dir / "samples.jsonl.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8") as f:
        for spec in cfg.passes():
            log.info("eval_pass_start", benchmark=benchmark, pass_name=spec.name, n=spec.n)
            records, stats = run_pass(gen, executor, problems, spec, cfg, benchmark)
            for r in records:
                f.write(json.dumps(r) + "\n")
            passes[spec.name] = stats

    metrics: dict[str, float | None] = {
        "pass@1_greedy": passes.get("greedy", {}).get("metrics", {}).get("pass@1"),
        "pass@1_sampled": passes.get("sampled", {}).get("metrics", {}).get("pass@1"),
        "pass@10_sampled": passes.get("sampled", {}).get("metrics", {}).get("pass@10"),
    }
    summary = {
        "model": cfg.model,
        "adapter": cfg.adapter,
        "benchmark": benchmark,
        "num_problems": len(problems),
        "limit": cfg.limit,
        "metrics": metrics,
        "passes": passes,
        "config": {
            "max_new_tokens": cfg.max_new_tokens,
            "seed": cfg.seed,
            "batch_size": cfg.batch_size,
            "system_prompt": SYSTEM_PROMPT,
            "correct_means": "sandbox status == passed and no integrity flags",
        },
        "generator": gen.info(),
        "sandbox": sandbox_info(executor),
        "env": environment(),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("eval_benchmark_done", benchmark=benchmark, out=str(out_dir), metrics=metrics)
    return summary


def run_eval(gen: Generator, cfg: EvalConfig, executor: Executor | None = None) -> list[dict]:
    """Evaluate ``gen`` on every benchmark in ``cfg``; returns the summaries."""
    own = executor is None
    ex = executor or get_executor(profile="eval")
    info = sandbox_info(ex)
    log.info("eval_sandbox", **info)
    if info["isolation"] not in ("bwrap", "docker"):
        exposed = [k for k in SECRET_ENV if os.environ.get(k)]
        log.warning(
            "eval_sandbox_weak_isolation",
            isolation=info["isolation"],
            secrets_in_env=exposed,
            hint="generated code can read this process' environment via /proc and may reach "
            "the network; keep secrets out of the eval process (log to W&B afterwards with "
            "`python -m coderl.eval.report RESULTS --wandb`)"
            if exposed
            else None,
        )
    try:
        summaries = [evaluate(gen, ex, cfg, b) for b in cfg.benchmarks]
    finally:
        if own:
            ex.close()
    from coderl.eval.report import log_to_wandb

    log_to_wandb(summaries)
    return summaries


# ---------------------------------------------------------------------- CLI
@app.command()
def main(
    model: Annotated[str, typer.Option(help="HF model id or local path.")],
    out: Annotated[Path, typer.Option(help="Output dir; one sub-dir per benchmark.")],
    backend: Annotated[str, typer.Option(help="vllm | hf | auto | mock")] = "auto",
    benchmarks: Annotated[str, typer.Option(help="Comma-separated.")] = "humaneval,mbpp_test",
    adapter: Annotated[str | None, typer.Option(help="LoRA adapter path.")] = None,
    limit: Annotated[int | None, typer.Option(help="First N problems per benchmark.")] = None,
    dtype: Annotated[str, typer.Option(help="auto | float16 | bfloat16 | float32")] = "auto",
    n_samples: Annotated[int, typer.Option(help="Samples per problem (sampled pass).")] = 20,
    temperature: Annotated[float, typer.Option()] = 0.8,
    top_p: Annotated[float, typer.Option()] = 0.95,
    max_new_tokens: Annotated[int, typer.Option()] = 512,
    seed: Annotated[int, typer.Option()] = 1234,
    batch_size: Annotated[int, typer.Option(help="Problems per batch.")] = 32,
    data_dir: Annotated[Path, typer.Option()] = Path("data/processed"),
    greedy: Annotated[bool, typer.Option("--greedy/--no-greedy")] = True,
    sampled: Annotated[bool, typer.Option("--sampled/--no-sampled")] = True,
    sandbox_backend: Annotated[
        str | None, typer.Option(help="subprocess | docker (default: platform default)")
    ] = None,
    sandbox_workers: Annotated[int | None, typer.Option()] = None,
    tensor_parallel_size: Annotated[int, typer.Option(help="vLLM tensor parallelism.")] = 1,
    gpu_memory_utilization: Annotated[float, typer.Option()] = 0.85,
    max_model_len: Annotated[int, typer.Option()] = 2048,
    enforce_eager: Annotated[bool, typer.Option(help="vLLM: disable CUDA graphs.")] = False,
    hf_batch_size: Annotated[int, typer.Option(help="HF: max sequences per forward.")] = 32,
    fallback_hf: Annotated[
        bool, typer.Option("--fallback-hf/--no-fallback-hf", help="Fall back to HF if vLLM fails.")
    ] = True,
) -> None:
    names = [b.strip() for b in benchmarks.split(",") if b.strip()]
    for b in names:
        if b not in BENCHMARKS:
            raise typer.BadParameter(f"unknown benchmark {b!r}; expected {list(BENCHMARKS)}")
    cfg = EvalConfig(
        model=model,
        out=out,
        benchmarks=names,
        adapter=adapter,
        limit=limit,
        data_dir=data_dir,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        seed=seed,
        batch_size=batch_size,
        greedy=greedy,
        sampled=sampled,
        sandbox_workers=sandbox_workers,
    )
    gen: Generator
    if backend == "mock":
        from coderl.gen.mock import MockGenerator, canonical_responder

        probs = [p for b in names for p in load_benchmark(b, data_dir, limit)]
        gen = MockGenerator(canonical_responder(probs), model_id=model)
    else:
        from coderl.gen import get_generator

        gen = get_generator(
            backend,
            model,
            dtype=dtype,
            adapter_path=adapter,
            fallback_to_hf=fallback_hf,
            vllm_kwargs={
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "enforce_eager": enforce_eager,
                "seed": seed,
            },
            hf_kwargs={"max_batch_size": hf_batch_size},
        )
    executor = get_executor(sandbox_backend, profile="eval")
    try:
        summaries = run_eval(gen, cfg, executor)
    finally:
        executor.close()
        gen.close()
    from coderl.eval.report import render_table

    sys.stdout.write(render_table(summaries) + "\n")


if __name__ == "__main__":
    app()
