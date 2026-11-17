.PHONY: install lint typecheck test test-fast test-docker data sandbox-image sandbox-cleanup bench-sandbox all

install:
	uv sync --extra dev

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
