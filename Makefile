.PHONY: server vllm test lint fmt help push-vm pull-vm cluster-eval

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
UNIT_TESTS ?= tests.test_gateway_paths tests.test_accel_timeslicer tests.test_trainer_optimizer_correctness tests.test_worker_manager tests.test_k8s_worker_manager tests.test_redis_store tests.test_cluster_eval_script tests.test_delta_weight_sync
# Only forward BASE_URL to e2e when the user supplied it. The Makefile default
# is for local CLI usage; e2e should start its own backend by default.
TRAINING_TEST_BASE_URL ?= $(if $(filter environment command line,$(origin BASE_URL)),$(BASE_URL),)
TRAINING_TEST_EXTRA ?= gpu
TRAINING_TEST_ARGS ?=
PIGLATIN_TEST_PYTHONPATH ?= examples/sft/pig-latin
EVAL_MODEL_PATH ?=
EVAL_EXAMPLES ?= 100
EVAL_DATA_PATH ?=
EVAL_NAMESPACE ?=
E2E_SCENARIO ?=
E2E_ARGS ?=
E2E_NAMESPACE ?=

# CUDA_VISIBLE_DEVICES can be provided either as an environment variable or as a
# Make variable, and is inherited by the backend/eval subprocesses.
ifneq ($(origin CUDA_VISIBLE_DEVICES),undefined)
  export CUDA_VISIBLE_DEVICES
endif

help:
	@echo "make server                              # $(BASE_MODEL), SAMPLING_BACKEND=$(SAMPLING_BACKEND), port $(PORT)"
	@echo "make server BASE_MODEL=google/gemma-4-e2b SAMPLING_BACKEND=vllm"
	@echo "VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM make vllm BASE_MODEL=google/gemma-4-e2b"
	@echo "make test                               # fast unit tests"
	@echo "make test e2e tiny-lora|tiny-fft|tiny-rl|lora-textsql|fft-gsm8k|fft-gsm8k-x2|fft-textsql-rl|fft-textsql-rl-x2  # tiny-* = fast overfit smoke tests"
	@echo "make test e2e tiny-lora BASE_URL=http://host:9003"
	@echo "CUDA_VISIBLE_DEVICES=0 make test e2e tiny-fft"
	@echo "make test e2e tiny-fft TRAINING_TEST_ARGS='steps=20'"
	@echo "make test e2e fft-gsm8k TRAINING_TEST_ARGS='steps=10 eval_examples=8 extra=\"batch=2\"'"
	@echo "make test piglatin                      # pig-latin example end-to-end tests"
	@echo "make cluster-eval EVAL_MODEL_PATH=/mnt/shared/open-rl/checkpoints/...  # one-off vLLM eval job on the cluster"
	@echo "make lint | fmt"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
server:
	@-kill -9 $$(lsof -ti:$(PORT)) 2>/dev/null || true
	BASE_MODEL="$(BASE_MODEL)" SAMPLING_BACKEND="$(SAMPLING_BACKEND)" \
	  uv run --extra $(if $(filter vllm,$(SAMPLING_BACKEND)),gpu,cpu) \
	  python -m uvicorn server.gateway:app --host $(HOST) --port $(PORT)

vllm:
	BASE_MODEL="$(BASE_MODEL)" \
	  uv run --extra vllm python -m server.vllm_sampler

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ifeq (cli,$(firstword $(MAKECMDGOALS)))
  CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(CLI_ARGS):;@:)
endif

ifeq (test,$(firstword $(MAKECMDGOALS)))
  TEST_MODE := $(word 2,$(MAKECMDGOALS))
  TEST_SCENARIO := $(word 3,$(MAKECMDGOALS))
  TEST_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifneq ($(TEST_ARGS),)
    $(eval $(TEST_ARGS):;@:)
  endif
endif

cli:
	@cd dev/tools && BASE_URL="$(BASE_URL)" uv run python cli.py $(CLI_ARGS)

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------
test:
	@mode="$(TEST_MODE)"; \
	scenario="$(TEST_SCENARIO)"; \
	if [ -z "$$mode" ] || [ "$$mode" = "unit" ]; then \
	  uv run --frozen --exact --extra cpu --extra cluster python -m unittest $(UNIT_TESTS); \
	elif [ "$$mode" = "e2e" ]; then \
	  if [ -z "$$scenario" ]; then \
	    echo "Missing e2e scenario. Expected tiny-lora, tiny-fft, tiny-rl, lora-textsql, fft-gsm8k, fft-gsm8k-x2, fft-textsql-rl, or fft-textsql-rl-x2."; \
	    exit 2; \
	  fi; \
	  set -- "scenario=$$scenario" "uv_extra=$(TRAINING_TEST_EXTRA)"; \
	  if [ -n "$(TRAINING_TEST_BASE_URL)" ]; then set -- "$$@" "base_url=$(TRAINING_TEST_BASE_URL)"; fi; \
	  kubectl delete pods -l accel-timeslicer=true --force --grace-period=0 2>/dev/null || true; \
	  uv run --extra "$(TRAINING_TEST_EXTRA)" python scripts/run_training_e2e.py "$$@" $(TRAINING_TEST_ARGS); \
	elif [ "$$mode" = "piglatin" ]; then \
	  PYTHONPATH="$(PIGLATIN_TEST_PYTHONPATH)" uv --project examples run python -m unittest tests.test_piglatin_qwen tests.test_piglatin_gemma; \
	else \
	  echo "Unknown test mode '$$mode'. Expected unit, e2e, or piglatin."; \
	  exit 2; \
	fi

lint:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .

fmt:
	uv run --extra dev ruff check --fix .
	uv run --extra dev ruff format .

# ---------------------------------------------------------------------------
# Deployment (GKE)
# ---------------------------------------------------------------------------
GCP_PROJECT ?= cdrollouts-sunilarora
IMAGE_TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || cat VERSION 2>/dev/null || echo latest)

build-images:
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) -f src/server/Dockerfile .
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) -f src/server/Dockerfile.gateway .
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-client:$(IMAGE_TAG) -f src/server/Dockerfile.client .

push-images:
	docker push gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG)
	docker push gcr.io/$(GCP_PROJECT)/open-rl-client:$(IMAGE_TAG)
	kubectl set image deployment/open-rl-gateway gateway=gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) 2>/dev/null || true
	kubectl set image daemonset/open-rl-accel-timeslicer accel-timeslicer=gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) 2>/dev/null || true
	kubectl set env deployment/open-rl-gateway OPEN_RL_WORKER_IMAGE=gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) 2>/dev/null || true

deploy:
	kubectl apply -k k8s/deploy/distributed-lustre/

# FFT DRA variant: the gateway launches one worker pod per FFT model, all pinned
# to one physical GPU allocation via a shared DRA ResourceClaim.
# See docs/setup/gke-fft-timeslice.md.
deploy-fft-timeslice:
	kubectl apply -k k8s/deploy/distributed-fft-timeslice/

rollout:
	kubectl rollout restart deployment redis-store open-rl-gateway open-rl-trainer-worker vllm-worker

# One-off vLLM eval of a checkpoint on the shared PVC:
cluster-eval:
	@if [ -z "$(EVAL_MODEL_PATH)" ]; then \
	  echo "Missing EVAL_MODEL_PATH. Example:"; \
	  echo "  make cluster-eval EVAL_MODEL_PATH=/mnt/shared/open-rl/checkpoints/<model-id>/weights/final"; \
	  exit 2; \
	fi; \
	set -- --model-path "$(EVAL_MODEL_PATH)" --examples "$(EVAL_EXAMPLES)"; \
	if [ -n "$(EVAL_DATA_PATH)" ]; then set -- "$$@" --data-path "$(EVAL_DATA_PATH)"; fi; \
	if [ -n "$(EVAL_NAMESPACE)" ]; then set -- "$$@" --namespace "$(EVAL_NAMESPACE)"; fi; \
	python3 scripts/run_cluster_eval.py "$$@"

# One-off E2E training/RL client job on the Kubernetes cluster:
cluster-e2e:
	@if [ -z "$(E2E_SCENARIO)" ]; then \
	  echo "Missing E2E_SCENARIO. Example:"; \
	  echo "  make cluster-e2e E2E_SCENARIO=fft-gsm8k-rl-x2 E2E_ARGS=\"base_model=Qwen/Qwen3-8B steps=30 jitter_sec=5\""; \
	  exit 2; \
	fi; \
	set -- --scenario "$(E2E_SCENARIO)" --image "gcr.io/$(GCP_PROJECT)/open-rl-client:$(IMAGE_TAG)"; \
	if [ -n "$(E2E_ARGS)" ]; then set -- "$$@" --args "$(E2E_ARGS)"; fi; \
	if [ -n "$(E2E_NAMESPACE)" ]; then set -- "$$@" --namespace "$(E2E_NAMESPACE)"; fi; \
	python3 scripts/run_cluster_e2e.py "$$@"

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
	@git rev-parse --short HEAD > VERSION 2>/dev/null || true
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'scratch' ./ $(REMOTE_HOST):~/open-rl

# Pull changes from the remote VM back to the local workspace
pull-vm:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'scratch' $(REMOTE_HOST):~/open-rl/ ./
