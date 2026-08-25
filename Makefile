.PHONY: server vllm test lint fmt help render release-bundle push-vm pull-vm

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
UNIT_TESTS ?= tests.test_gateway_paths tests.test_snapshot_agent tests.test_trainer_optimizer_correctness tests.test_worker_launch_processor
# Only forward BASE_URL to e2e when the user supplied it. The Makefile default
# is for local CLI usage; e2e should start its own backend by default.
TRAINING_TEST_BASE_URL ?= $(if $(filter environment command line,$(origin BASE_URL)),$(BASE_URL),)
TRAINING_TEST_EXTRA ?= gpu
TRAINING_TEST_ARGS ?=
PIGLATIN_TEST_PYTHONPATH ?= examples/sft/pig-latin

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
	@echo "make test e2e tiny-lora|tiny-fft|tiny-rl|lora-textsql|fft-gsm8k|fft-gsm8k-x2  # tiny-* = fast overfit smoke tests"
	@echo "make test e2e tiny-lora BASE_URL=http://host:9003"
	@echo "CUDA_VISIBLE_DEVICES=0 make test e2e tiny-fft"
	@echo "make test e2e tiny-fft TRAINING_TEST_ARGS='steps=20'"
	@echo "make test e2e fft-gsm8k TRAINING_TEST_ARGS='steps=10 eval_examples=8 extra=\"batch=2\"'"
	@echo "make test piglatin                      # pig-latin example end-to-end tests"
	@echo "make lint | fmt"
	@echo "make render OVERLAY=k8s/deploy/distributed-shared VERSION=v0.0.1  # pinned manifests to stdout"
	@echo "make release-bundle VERSION=v0.0.1     # release assets into $(DIST_DIR)/"

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
	  uv run --frozen --exact --extra cpu python -m unittest $(UNIT_TESTS); \
	elif [ "$$mode" = "e2e" ]; then \
	  if [ -z "$$scenario" ]; then \
	    echo "Missing e2e scenario. Expected tiny-lora, tiny-fft, tiny-rl, lora-textsql, fft-gsm8k, or fft-gsm8k-x2."; \
	    exit 2; \
	  fi; \
	  set -- "scenario=$$scenario" "uv_extra=$(TRAINING_TEST_EXTRA)"; \
	  if [ -n "$(TRAINING_TEST_BASE_URL)" ]; then set -- "$$@" "base_url=$(TRAINING_TEST_BASE_URL)"; fi; \
	  uv run --extra "$(TRAINING_TEST_EXTRA)" python scripts/run_training_e2e.py "$$@" $(TRAINING_TEST_ARGS); \
	elif [ "$$mode" = "piglatin" ]; then \
	  PYTHONPATH="$(PIGLATIN_TEST_PYTHONPATH)" uv --project examples run python -m unittest discover -s tests; \
	else \
	  echo "Unknown test mode '$$mode'. Expected unit, e2e, or piglatin."; \
	  exit 2; \
	fi

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
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-server:$(IMAGE_TAG) -f src/server/Dockerfile .
	DOCKER_BUILDKIT=1 docker build -t gcr.io/$(GCP_PROJECT)/open-rl-gateway:$(IMAGE_TAG) -f src/server/Dockerfile.gateway .

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
# Release
# ---------------------------------------------------------------------------
# Images published by .github/workflows/build-and-push.yml. The names must match
# the manifests exactly or the pin silently does nothing.
IMAGE_REPO     ?= ghcr.io/gke-labs/open-rl
RELEASE_IMAGES ?= server gateway client
# Gitignored; the release job uploads everything in here.
DIST_DIR       ?= dist

# Print any overlay with the open-rl images pinned to VERSION:
#   make render OVERLAY=examples/text-to-sql VERSION=v0.0.1 | kubectl apply -f -
# Works on a temp copy of the repo: `kustomize edit` rewrites the kustomization
# in place, and overlays reach into k8s/deploy/ by relative path.
render:
	@test -n "$(OVERLAY)" && test -d "$(OVERLAY)" || { echo "Set OVERLAY=<dir> to an overlay directory"; exit 2; }
	@test -n "$(VERSION)" || { echo "Set VERSION=<tag>, e.g. VERSION=v0.0.1"; exit 2; }
	@set -eu; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT INT TERM; \
	tar -cf - --exclude .git --exclude .venv --exclude __pycache__ --exclude $(DIST_DIR) . | tar -xf - -C "$$tmp"; \
	cd "$$tmp/$(OVERLAY)"; \
	for image in $(RELEASE_IMAGES); do \
	  kustomize edit set image "$(IMAGE_REPO)/$$image:$(VERSION)"; \
	done; \
	kustomize build .

# Build the assets attached to a GitHub Release. Asset names carry no version so
# that /releases/latest/download/<name> keeps resolving.
release-bundle:
	@test -n "$(VERSION)" || { echo "Set VERSION=<tag>, e.g. VERSION=v0.0.1"; exit 2; }
	@rm -rf $(DIST_DIR) && mkdir -p $(DIST_DIR)
	@$(MAKE) --no-print-directory render OVERLAY=k8s/deploy/distributed-shared VERSION=$(VERSION) > $(DIST_DIR)/openrl-distributed-shared.yaml
	@$(MAKE) --no-print-directory render OVERLAY=k8s/deploy/distributed-lustre VERSION=$(VERSION) > $(DIST_DIR)/openrl-distributed-lustre.yaml
	@cd $(DIST_DIR) && { command -v sha256sum >/dev/null && sha256sum *.yaml || shasum -a 256 *.yaml; } > checksums.sha256

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
