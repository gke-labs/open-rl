.PHONY: server vllm test lint fmt help

# ---------------------------------------------------------------------------
# Knobs (override on the command line: make server BASE_MODEL=... SAMPLER=...)
# ---------------------------------------------------------------------------
# The HuggingFace base model checkpoint loaded by the server and training workers
BASE_MODEL     ?= Qwen/Qwen3-0.6B
# The backend used for sampling ("torch" for local inference, or "vllm" for optimized remote inference)
SAMPLER        ?= torch
# Whether to run the API gateway and training worker loop together in a single process (1=yes, 0=no)
SINGLE_PROCESS ?= 1
# The network interface to bind the API server
HOST           ?= 127.0.0.1
# The local port number for the API server
PORT           ?= 9003
# The fully qualified base URL used by local CLI tools and clients
BASE_URL       ?= http://$(HOST):$(PORT)
TEST_PYTHONPATH ?= examples/sft/pig-latin

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
	cd src/server && SINGLE_PROCESS="$(SINGLE_PROCESS)" BASE_MODEL="$(BASE_MODEL)" SAMPLER="$(SAMPLER)" \
	  uv run --extra $(if $(filter vllm,$(SAMPLER)),gpu,cpu) \
	  python -m uvicorn gateway:app --host $(HOST) --port $(PORT)

vllm:
	cd src/server && BASE_MODEL="$(BASE_MODEL)" \
	  uv run --extra vllm python -m vllm_sampler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ifeq (cli,$(firstword $(MAKECMDGOALS)))
  CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(CLI_ARGS):;@:)
endif

cli:
	@cd dev/tools && BASE_URL="$(BASE_URL)" uv run python cli.py $(CLI_ARGS)

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
test:
	PYTHONPATH="$(TEST_PYTHONPATH)" uv --project examples run python -m unittest discover -s tests

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
	cd src/server && DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) -f Dockerfile .
	cd src/server && DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) -f Dockerfile.gateway .

push-images:
	docker push gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG)

deploy:
	kubectl apply -k k8s/deploy/distributed-lustre/

rollout:
	kubectl rollout restart deployment redis-store open-rl-gateway open-rl-trainer-worker vllm-worker

# Local Redis (for testing distributed mode):
#   sudo apt install redis-server && sudo service redis-server start
#   redis-cli ping   # should print PONG
#   sudo service redis-server stop

# GKE client jobs — run directly:
#   kubectl apply -f examples/rl/rlvr/rlvr-job.yaml
#   kubectl apply -f examples/rl/tinker-rl-basic/tinker-rl-basic-job.yaml
#   kubectl logs -f job/<job-name>
#   kubectl delete job <job-name>

dashboard-apply:
	@dev/monitoring/apply_dashboard.sh $(GCP_PROJECT)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
REMOTE_HOST ?= mars

server-sync:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ./ $(REMOTE_HOST):~/work/open-rl
