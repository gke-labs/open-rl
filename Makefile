.PHONY: server vllm test lint fmt help

# ---------------------------------------------------------------------------
# Knobs (override on the command line: make server BASE_MODEL=... SAMPLER=...)
# ---------------------------------------------------------------------------
BASE_MODEL ?= Qwen/Qwen3-0.6B
SAMPLER    ?= torch
PORT       ?= 9003

help:
	@echo "make server                              # $(BASE_MODEL), SAMPLER=$(SAMPLER), port $(PORT)"
	@echo "make server BASE_MODEL=google/gemma-4-e2b SAMPLER=vllm"
	@echo "make vllm   BASE_MODEL=google/gemma-4-e2b  # standalone vLLM worker"
	@echo "make test | lint | fmt"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
server:
	@-kill -9 $$(lsof -ti:$(PORT)) 2>/dev/null || true
	cd server && SINGLE_PROCESS=1 BASE_MODEL="$(BASE_MODEL)" SAMPLER="$(SAMPLER)" \
	  uv run --extra $(if $(filter vllm,$(SAMPLER)),gpu,cpu) \
	  python -m uvicorn src.gateway:app --host 127.0.0.1 --port $(PORT)

vllm:
	cd server && BASE_MODEL="$(BASE_MODEL)" \
	  uv run --extra vllm python -m src.vllm_sampler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ifeq (cli,$(firstword $(MAKECMDGOALS)))
  CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(CLI_ARGS):;@:)
endif

cli:
	@cd client && uv run python cli.py $(CLI_ARGS)

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
test:
	cd client && uv run python -m unittest discover tests

lint:
	uvx ruff check .
	uvx ruff format --check .

fmt:
	uvx ruff check --fix .
	uvx ruff format .

# ---------------------------------------------------------------------------
# Deployment (GKE)
# ---------------------------------------------------------------------------
GCP_PROJECT ?= cdrollouts-sunilarora
IMAGE_TAG   ?= latest

build-images:
	cd server && DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) -f Dockerfile .
	cd server && DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) -f Dockerfile.gateway .

push-images:
	docker push gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG)

deploy:
	kubectl apply -k server/kubernetes/distributed-lustre/

rollout:
	kubectl rollout restart deployment redis-store open-rl-gateway open-rl-trainer-worker vllm-worker

# Local Redis (for testing distributed mode):
#   sudo apt install redis-server && sudo service redis-server start
#   redis-cli ping   # should print PONG
#   sudo service redis-server stop

# GKE client jobs — run directly:
#   kubectl apply -f client/kubernetes/rlvr-job.yaml
#   kubectl apply -f client/kubernetes/tinker-rl-basic-job.yaml   (or -2, -3)
#   kubectl logs -f job/<job-name>
#   kubectl delete job <job-name>

dashboard-apply:
	@scripts/apply_dashboard.sh $(GCP_PROJECT)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
HOST ?= mars

server-sync:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ./ $(HOST):~/work/open-rl

diagrams:
	zsh -ic "mmdc -i assets/architecture.mmd -o assets/architecture.svg -b transparent"
	zsh -ic "mmdc -i assets/architecture.mmd -o assets/architecture.png -s 3 -b transparent"
