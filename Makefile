#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = NN-japanese-speaker
PYTHON_VERSION = 3.11
SRC := main.py japanese_speaker_recognition tests

# Ruff options
RUFF_CHECK_OPTS := --respect-gitignore
RUFF_FORMAT_OPTS :=

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make install        - sync runtime deps with uv (creates .venv)"
	@echo "  make install-dev    - sync runtime + dev deps (ruff/mypy/pre-commit)"
	@echo "  make lint           - ruff check + format --check"
	@echo "  make format         - ruff auto-fix + format"
	@echo "  make typecheck      - mypy"
	@echo "  make check          - lint + typecheck"
	@echo "  make pre-commit     - install & run hooks"
	@echo "  make clean          - remove __pycache__/pyc"

## Install runtime deps into .venv (pins from pyproject/uv.lock)
.PHONY: install
install:
	uv python install $(PYTHON_VERSION)
	uv sync

## Install runtime + dev deps (ruff/mypy/pre-commit pinned below)
.PHONY: install-dev
install-dev:
	uv python install $(PYTHON_VERSION)
	uv sync --dev

## Lint using ruff (no modifications)
.PHONY: lint
lint:
	uv run ruff format --check $(RUFF_FORMAT_OPTS) $(SRC)
	uv run ruff check $(RUFF_CHECK_OPTS) $(SRC)

## Auto-fix with ruff
.PHONY: format
format:
	uv run ruff check --fix $(RUFF_CHECK_OPTS) $(SRC)
	uv run ruff format $(RUFF_FORMAT_OPTS) $(SRC)

## Static type checking
.PHONY: typecheck
typecheck:
	uv run mypy $(SRC)

## All checks
.PHONY: check
check: lint typecheck

.PHONY: install-tools
install-tools:
	uv add pre-commit
	uv add ruff
	uv add mypy

## Pre-commit (installed in the .venv, runs with uv)
.PHONY: pre-commit
pre-commit:
	uv run pre-commit install
	uv run pre-commit run --all-files

## Clean caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
