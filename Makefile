.PHONY: sync format lint test check list

sync:
	uv sync --extra cu128

format:
	uv run ruff format
	uv run ruff check --fix

lint:
	uv run ruff format --check
	uv run ruff check

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

check: lint test

list:
	uv run list-envs
