# docs-rag-evals — one-command workflows. Run `make help`.
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Docker Compose v2 plugin ("docker compose") or the standalone binary ("docker-compose").
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
HOST ?= 127.0.0.1
PORT ?= 8000
RUN := uv run

.PHONY: help setup up down nuke ingest serve demo test test-integration lint eval eval-full eval-rerankers check-key \
        golden-check golden-candidates corpus

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv with Python 3.12 + locked deps (uv)
	uv sync

up: ## Start Postgres 16 + pgvector (docker compose, host port 5433 by default)
	$(COMPOSE) up -d --wait db

down: ## Stop the database (keeps the pgdata volume)
	$(COMPOSE) down

nuke: ## Stop the database AND delete its volume
	$(COMPOSE) down -v

ingest: up ## Migrate, chunk and embed the corpus (incremental; no-op when up to date)
	$(RUN) python -m ragkit.ingest

serve: ## Run the API + UI (default http://127.0.0.1:8000)
	$(RUN) uvicorn ragkit.api:app --host $(HOST) --port $(PORT)

demo: setup ingest ## One command: db up, ingest if needed, serve the UI
	@echo ""
	@echo "  ➜  Open http://$(HOST):$(PORT)   (Ctrl+C to stop; 'make down' stops the db)"
	@echo ""
	$(RUN) uvicorn ragkit.api:app --host $(HOST) --port $(PORT)

test: ## Unit + API tests (no db, no model downloads, no API key)
	$(RUN) pytest

test-integration: ingest ## Tests against the real Postgres index and models
	RUN_INTEGRATION=1 $(RUN) pytest -m integration

lint: ## Ruff lint + format check
	$(RUN) ruff check ragkit evals tests scripts
	$(RUN) ruff format --check ragkit evals tests scripts

golden-check: ## Sanity-check evals/golden.jsonl against the corpus
	$(RUN) python evals/check_golden.py

eval: ingest golden-check ## Offline retrieval ablation -> EVAL_REPORT.md (no API key needed)
	$(RUN) python evals/run.py

check-key:
	@$(RUN) python -c "import sys; from ragkit.config import get_settings, NO_KEY_MESSAGE; s = get_settings(); s.llm_enabled or sys.exit('ERROR: ' + NO_KEY_MESSAGE)"

eval-full: check-key ingest golden-check ## Retrieval + generation metrics with Claude (needs ANTHROPIC_API_KEY)
	$(RUN) python evals/run.py --full

eval-rerankers: ingest ## Compare the default cross-encoder with BAAI/bge-reranker-base (~1.1 GB download)
	$(RUN) python evals/run.py --extra-rerankers BAAI/bge-reranker-base

golden-candidates: check-key ## Ask Claude to propose new golden questions for human review (needs key)
	$(RUN) python scripts/generate_golden.py

corpus: ## Re-download and rebuild the vendored corpus/ from the pinned FastAPI tag
	$(RUN) python scripts/fetch_corpus.py
