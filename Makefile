.PHONY: install lint typecheck test data all

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy coderl

test:
	uv run pytest --cov=coderl --cov-report=term-missing

data:
	uv run python -m coderl.data.build --out data/processed

all: lint typecheck test
