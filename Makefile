# Task runner for arxiv-reproducer. `make help` lists targets.
VENV ?= .venv/bin
IMAGE ?= arxiv-repro-sandbox:latest

.PHONY: help install test integration lint typecheck check build-image lock run paper

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install: ## Editable install with dev + locked dependencies
	$(VENV)/pip install -r requirements-lock.txt
	$(VENV)/pip install -e . --no-deps

test: ## Unit tests with coverage floor
	$(VENV)/pytest -q

integration: ## Real-Docker integration tests (builds the sandbox image)
	$(VENV)/pytest -q --run-docker -k RealDocker --no-cov

lint: ## Ruff
	$(VENV)/ruff check src tests

typecheck: ## Mypy over src
	$(VENV)/mypy src

check: lint typecheck test ## Everything CI runs

build-image: ## Build the pre-baked sandbox image
	docker build -t $(IMAGE) -f src/arxiv_reproducer/docker/sandbox.Dockerfile .

lock: ## Regenerate requirements-lock.txt (needs uv)
	uv pip compile --universal --python-version 3.11 --extra dev -o requirements-lock.txt pyproject.toml

run: ## Run a reproduction: make run PAPER=2301.12345
	$(VENV)/arxiv-repro $(PAPER)

paper: ## Build paper/main.pdf (tectonic if available, else pdflatex x2)
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic paper/main.tex; \
	else \
		cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
		&& pdflatex -interaction=nonstopmode -halt-on-error main.tex; \
	fi
