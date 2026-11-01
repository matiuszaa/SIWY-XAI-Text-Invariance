.PHONY: install lint format test smoke run

CONFIG ?= configs/small.yaml

install:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

smoke:
	uv run xai-inv run --config configs/smoke.yaml

run:
	uv run xai-inv run --config $(CONFIG)
