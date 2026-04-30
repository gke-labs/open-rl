.PHONY: server vllm test lint fmt help push-vm pull-vm

# ---------------------------------------------------------------------------
# Knobs (override on the command line: make server BASE_MODEL=... SAMPLING_BACKEND=...)
# ---------------------------------------------------------------------------
# The HuggingFace base model checkpoint loaded by the server and training workers
BASE_MODEL     ?= google/gemma-4-e2b
# The backend used for sampling ("torch" for local inference, or "vllm" for optimized remote inference)
SAMPLING_BACKEND ?= torch
# The network interface to bind the API server
HOST           ?= 127.0.0.1
# The local port number for the API server
PORT           ?= 9003
# The fully qualified base URL used by local CLI tools and clients
BASE_URL       ?= http://$(HOST):$(PORT)
TEST_PYTHONPATH ?= examples/sft/pig-latin

help:
	@echo "make server                              # $(BASE_MODEL), SAMPLING_BACKEND=$(SAMPLING_BACKEND), port $(PORT)"
	@echo "make server BASE_MODEL=google/gemma-4-e2b SAMPLING_BACKEND=vllm"
	@echo "VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM make vllm BASE_MODEL=google/gemma-4-e2b"
	@echo "make test | lint | fmt"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
server:
	@-kill -9 $$(lsof -ti:$(PORT)) 2>/dev/null || true
	cd src/server && BASE_MODEL="$(BASE_MODEL)" SAMPLING_BACKEND="$(SAMPLING_BACKEND)" \
	  uv run --extra $(if $(filter vllm,$(SAMPLING_BACKEND)),gpu,cpu) \
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
# Remote host address for VM synchronization. Override on command line: make push-vm REMOTE_HOST=...
REMOTE_HOST ?= <PLACE_HOLDER_FOR_REMOTE_HOST_ADDRESS>

# Push local workspace changes to the remote VM
push-vm:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ./ $(REMOTE_HOST):~/open-rl

# Pull changes from the remote VM back to the local workspace
pull-vm:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' $(REMOTE_HOST):~/open-rl/ ./

