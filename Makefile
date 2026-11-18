.PHONY: all install run debug test clean lint lint-strict

USER ?= $(shell id -un)
VENV ?= .venv
GOINFRE ?= $(shell [ -d "/goinfre/$(USER)" ] && echo "/goinfre/$(USER)" || ([ -d "/home/$(USER)/goinfre" ] && echo "/home/$(USER)/goinfre" || ([ -d "/goinfre" ] && echo "/goinfre/$(USER)" || echo "$(HOME)/.cache")))
VENV_TARGET ?= $(GOINFRE)/RAG_venv

export UV_PROJECT_ENVIRONMENT ?= $(VENV_TARGET)
export UV_CACHE_DIR ?= $(GOINFRE)/uv-cache
export HF_HOME ?= $(GOINFRE)/huggingface
export TRANSFORMERS_CACHE ?= $(GOINFRE)/huggingface/transformers
export TORCH_HOME ?= $(GOINFRE)/torch

all: install

install:
	@echo "Configuring cache and environment in goinfre..."
	@mkdir -p "$(UV_CACHE_DIR)" "$(HF_HOME)" "$(TORCH_HOME)"
	@if [ -d "$(GOINFRE)" ]; then \
		mkdir -p "$(VENV_TARGET)"; \
		if [ ! -L "$(VENV)" ] || [ "$$(readlink -f $(VENV))" != "$$(readlink -f $(VENV_TARGET))" ]; then \
			echo "Redirecting $(VENV) -> $(VENV_TARGET)"; \
			rm -rf "$(VENV)"; \
			ln -sfn "$(VENV_TARGET)" $(VENV); \
		fi; \
		echo "Cache and virtual environment active in: $(GOINFRE)"; \
		df -h "$(GOINFRE)"; \
	fi
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	uv run pytest

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

fclean: clean
	rm -rf $(VENV)

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
