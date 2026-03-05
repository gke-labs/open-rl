.PHONY: run-server run-server-engine-sampler run-function-gemma-server run-function-gemma-sft run-sft run-sft-parallel run-rlvr run-rlvr-parallel

# Default VLLM model for inference, can be overridden via `make run-vllm VLLM_MODEL=...`
#VLLM_MODEL ?= Qwen/Qwen2.5-0.5B
VLLM_MODEL ?= Qwen/Qwen3-4B-Instruct-2507

# Default GPU allocations for running isolated processes locally/on VMs
TRAINER_GPU ?= 0
VLLM_GPU ?= 1

# Run the PyTorch Uvicorn Training Server locally
run-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(TRAINER_GPU)" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Kill any local process stuck listening on port 8000
kill-server:
	@kill -9 $$(lsof -ti:8000) 2>/dev/null || echo "Port 8000 is free"

# Run the standalone vLLM inference worker locally
run-vllm:
	cd server && UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(VLLM_GPU)" VLLM_MODEL="$(VLLM_MODEL)" uv run python -m src.vllm_worker

run-server-engine-sampler:
	cd server && UV_INDEX_URL="https://pypi.org/simple" SAMPLER_BACKEND=engine VLLM_MODEL="$(VLLM_MODEL)" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

run-function-gemma-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" SAMPLER_BACKEND=engine VLLM_MODEL="google/functiongemma-270m-it" uv run uvicorn src.main:app --host 127.0.0.1 --port 9000

run-function-gemma-sft:
	cd client && uv run --python 3.12 --no-sync -i https://pypi.org/simple python functiongemma_sft.py $(ARGS)

# Client test targets
run-sft:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --base-model "$(VLLM_MODEL)" $(ARGS)

run-sft-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --parallel --base-model "$(VLLM_MODEL)"

# Default concurrent jobs for parallel execution
JOBS ?= 2
STEPS ?= 15

run-rlvr:
	cd client && uv run --no-sync -i https://pypi.org/simple python rlvr.py --jobs 1 --steps $(STEPS) --base-model "$(VLLM_MODEL)"

run-rlvr-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python rlvr.py --jobs $(JOBS) --steps $(STEPS) --base-model "$(VLLM_MODEL)"

# Plot metrics from a JSONL file
# Usage: make plot-metrics [FILE=path/to/metrics.jsonl]
plot-metrics:
	cd client && uv run --no-sync -i https://pypi.org/simple python plot_metrics.py $(FILE)

# Plot parallel metrics from the RLVR log file
# Usage: make plot-logs [LOG_FILE=client/rlvr_parallel_results.log] [WATCH=1]
plot-logs:
	cd client && uv run --no-sync -i https://pypi.org/simple python plot_logs.py $(or $(LOG_FILE),rlvr_parallel_results.log) $(if $(WATCH),--watch,)

# Generate diagrams using local mmdc zsh alias
diagrams:
	zsh -ic "mmdc -i design_arch.mmd -o design_arch.svg"
	zsh -ic "mmdc -i rollout_flow.mmd -o rollout_flow.svg"
	zsh -ic "mmdc -i distributed_arch.mmd -o distributed_arch.svg"

HOST ?= b3

# Sync server to remote host $(HOST)
# TODO: sync only server directory
server-sync:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ./ $(HOST):~/work/open-rl

server-tunnel:
	ssh -fN -L 8000:localhost:8000 $(HOST)

# CLI Targets
# Usage: make run-cli list OR make run-cli chat --model ...
# This hack allows passing arguments directly after the target name
ifeq (run-cli,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run-cli"
  CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets so make doesn't complain
  $(eval $(CLI_ARGS):;@:)
endif

run-cli:
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py $(CLI_ARGS)

# Shortcut: make run-cli-list
run-cli-list:
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py list

# Shortcut: make run-cli-chat MODEL=... [PROMPT="..."]
run-cli-chat:
	@test -n "$(MODEL)" || (echo "Error: MODEL argument is required. Usage: make run-cli-chat MODEL=<model_id>" && exit 1)
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py chat --model $(MODEL) --system-prompt "$(or $(PROMPT),You are helpful geography assistant.)"

# --- Deployment Targets ---

GCP_PROJECT ?= cdrollouts-sunilarora
GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-server
IMAGE_TAG ?= latest

remote-build-setup:
	@echo "--- Setting up Remote Builder ($(HOST)) ---"
	ssh $(HOST) "gcloud auth configure-docker -q"
	@echo "--- Setup Complete! ---"

remote-build: server-sync
	@echo "--- Building Docker Image on $(HOST) ---"
	ssh $(HOST) "cd ~/work/open-rl/server && DOCKER_BUILDKIT=1 docker build -t $(GCR_REPO):$(IMAGE_TAG) ."

remote-push:
	@echo "--- Pushing Image to GCR from $(HOST) ---"
	ssh $(HOST) "docker push $(GCR_REPO):$(IMAGE_TAG)"


deploy:
	@echo "--- Deploying to GKE ---"
	kubectl apply -f server/kubernetes/

# --- Redis Management (Linux) ---

.PHONY: install-redis start-redis stop-redis

# Install redis-server if not present (assuming Debian/Ubuntu)
install-redis:
	@echo "--- Checking/Installing Redis Server ---"
	@if ! command -v redis-server >/dev/null 2>&1; then \
		echo "Redis not found. Installing via apt-get..."; \
		sudo apt-get update && sudo apt-get install -y redis-server; \
	else \
		echo "Redis is already installed."; \
	fi

# Start the local redis service
start-redis:
	@echo "--- Starting Redis Server ---"
	sudo service redis-server start
	@echo "Redis started. Check with: redis-cli ping"

# Stop the local redis service
stop-redis:
	@echo "--- Stopping Redis Server ---"
	sudo service redis-server stop
