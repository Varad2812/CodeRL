.PHONY: install install-hf smoke-eval report lint typecheck test test-fast test-docker data sandbox-image sandbox-cleanup bench-sandbox all

install:
	uv sync --extra dev

install-hf:  # torch + transformers + accelerate (macOS arm64 / CPU / CUDA); vllm stays Linux-only
	uv sync --extra dev --extra hf

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy coderl

test:
	uv run pytest --cov=coderl --cov-report=term-missing

test-fast:  # everything except tests that need Docker
	uv run pytest -m "not docker"

test-docker:  # sandbox Docker backend: security suite + canonical solutions
	uv run pytest -m docker tests/sandbox

data:
	uv run python -m coderl.data.build --out data/processed

sandbox-image:
	uv run python -m coderl.sandbox build-image

sandbox-cleanup:
	uv run python -m coderl.sandbox cleanup --all

bench-sandbox:
	uv run python scripts/bench_sandbox.py

all: lint typecheck test

SMOKE_MODEL ?= Qwen/Qwen2.5-Coder-0.5B-Instruct

smoke-eval:  # HF generator on MPS/CPU, 5 problems per benchmark, n=2, Docker sandbox
	CODERL_SANDBOX_BACKEND=docker uv run --extra dev --extra hf python -m coderl.eval.run_benchmarks \
		--model $(SMOKE_MODEL) --backend hf --benchmarks humaneval,mbpp_test \
		--out results/smoke/qwen0.5b --limit 5 --n-samples 2 --batch-size 5
	uv run python -m coderl.eval.report results/smoke

report:
	uv run python -m coderl.eval.report results/
