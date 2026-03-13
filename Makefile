.PHONY: run-server run-server-engine-sampler run-function-gemma-server run-function-gemma-sft run-pig-latin-server run-pig-latin-sft run-sft run-sft-parallel run-rlvr run-rlvr-parallel

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
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" SAMPLER_BACKEND=engine VLLM_MODEL="$(VLLM_MODEL)" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

run-function-gemma-server:
	cd server && OPEN_RL_SINGLE_PROCESS=1 SAMPLER_BACKEND=engine OPEN_RL_BASE_MODEL="google/functiongemma-270m-it" PYTHONUNBUFFERED=1 uv run --extra ml uvicorn src.main:app --host 127.0.0.1 --port 9000 $(ARGS)

run-function-gemma:
	cd client && uv run --python 3.12 functiongemma-demo $(ARGS)

run-pig-latin-server:
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" OPEN_RL_SINGLE_PROCESS=1 OPEN_RL_BASE_MODEL="Qwen/Qwen3-0.6B" SAMPLER_BACKEND=engine VLLM_MODEL="Qwen/Qwen3-0.6B" uv run uvicorn src.main:app --host 127.0.0.1 --port 9001

run-pig-latin-sft:
	cd client && uv run --python 3.12 --no-sync -i https://pypi.org/simple python -u piglatin_sft.py qwen $(ARGS)

run-pig-latin-gemma-server:
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" OPEN_RL_SINGLE_PROCESS=1 OPEN_RL_BASE_MODEL="google/gemma-3-1b-it" SAMPLER_BACKEND=engine VLLM_MODEL="google/gemma-3-1b-it" uv run --extra ml uvicorn src.main:app --host 127.0.0.1 --port 9002

run-pig-latin-gemma-sft:
	cd client && uv run --python 3.12 --no-sync -i https://pypi.org/simple python -u piglatin_sft.py gemma base_url="http://127.0.0.1:9002" $(ARGS)

# Client test targets
run-sft:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --base-model "$(VLLM_MODEL)" $(ARGS)

run-sft-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --parallel --base-model "$(VLLM_MODEL)"

# Default concurrent jobs for parallel execution
JOBS ?= 2
STEPS ?= 15

# OpenTelemetry Tracing Toggles (0 = disabled, 1 = enabled)
ENABLE_GCP_TRACE ?= 0
ENABLE_CONSOLE_TRACE ?= 0

run-rlvr:
	cd client && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) ENABLE_CONSOLE_TRACE=$(ENABLE_CONSOLE_TRACE) uv run --no-sync -i https://pypi.org/simple python rlvr.py --jobs 1 --steps $(STEPS) --base-model "$(VLLM_MODEL)"

run-rlvr-parallel:
	cd client && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) ENABLE_CONSOLE_TRACE=$(ENABLE_CONSOLE_TRACE) uv run --no-sync -i https://pypi.org/simple python rlvr.py --jobs $(JOBS) --steps $(STEPS) --base-model "$(VLLM_MODEL)"

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
	zsh -ic "mmdc -i design_arch.mmd -o design_arch.svg -b transparent"
	zsh -ic "mmdc -i rollout_flow.mmd -o rollout_flow.svg -b transparent"
	zsh -ic "mmdc -i distributed_arch.mmd -o distributed_arch.svg -b transparent"
	zsh -ic "mmdc -i architecture.mmd -o openrl_architecture.svg -b transparent"
	zsh -ic "mmdc -i architecture.mmd -o openrl_architecture.png -s 3 -b transparent"

HOST ?= mars

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
GATEWAY_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-gateway
CLIENT_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-client
TINKER_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/tinker-cookbook
IMAGE_TAG ?= latest

remote-build-setup:
	@echo "--- Setting up Remote Builder ($(HOST)) ---"
	ssh $(HOST) "gcloud auth configure-docker -q"
	@echo "--- Setup Complete! ---"

remote-build: server-sync
	@echo "--- Building Server Docker Images on $(HOST) ---"
	ssh $(HOST) "cd ~/work/open-rl/server && DOCKER_BUILDKIT=1 docker build -t $(GCR_REPO):$(IMAGE_TAG) -f Dockerfile ."
	ssh $(HOST) "cd ~/work/open-rl/server && DOCKER_BUILDKIT=1 docker build -t $(GATEWAY_GCR_REPO):$(IMAGE_TAG) -f Dockerfile.gateway ."

remote-push:
	@echo "--- Pushing Server Images to GCR from $(HOST) ---"
	ssh $(HOST) "docker push $(GCR_REPO):$(IMAGE_TAG)"
	ssh $(HOST) "docker push $(GATEWAY_GCR_REPO):$(IMAGE_TAG)"

remote-client-build: server-sync
	@echo "--- Building Client Docker Image on $(HOST) ---"
	ssh $(HOST) "cd ~/work/open-rl/client && DOCKER_BUILDKIT=1 docker build -t $(CLIENT_GCR_REPO):$(IMAGE_TAG) ."

remote-client-push:
	@echo "--- Pushing Client Image to GCR from $(HOST) ---"
	ssh $(HOST) "docker push $(CLIENT_GCR_REPO):$(IMAGE_TAG)"

tinker-sync:
	@echo "--- Syncing tinker-cookbook to $(HOST) ---"
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ../tinker-cookbook/ $(HOST):~/work/tinker-cookbook

remote-tinker-build: tinker-sync
	@echo "--- Building tinker-cookbook Docker Image on $(HOST) ---"
	ssh $(HOST) "cd ~/work/tinker-cookbook && DOCKER_BUILDKIT=1 docker build -t $(TINKER_GCR_REPO):$(IMAGE_TAG) ."

remote-tinker-push:
	@echo "--- Pushing tinker-cookbook Image to GCR from $(HOST) ---"
	ssh $(HOST) "docker push $(TINKER_GCR_REPO):$(IMAGE_TAG)"

deploy:
	@echo "--- Deploying Server to GKE ---"
	kubectl apply -f server/kubernetes/

run-client-job:
	@echo "--- Deploying RLVR Client Job to GKE ---"
	kubectl delete job open-rl-client-job --ignore-not-found=true
	kubectl apply -f client/kubernetes/rlvr-job.yaml
	@echo "Waiting for job to start..."
	@sleep 4
	kubectl logs -f job/open-rl-client-job

stop-client-job:
	@echo "--- Stopping RLVR Client Job ---"
	kubectl delete job open-rl-client-job open-rl-client-job-parallel --ignore-not-found=true

logs-client-job:
	@echo "--- Fetching RLVR Client Job Logs ---"
	kubectl logs -f job/open-rl-client-job

rollout:
	@echo "--- Rolling out latest server deployments ---"
	kubectl rollout restart deployment redis-broker open-rl-gateway open-rl-trainer-worker vllm-worker
	kubectl rollout status deployment open-rl-gateway
	kubectl rollout status deployment open-rl-trainer-worker

run-tinker-job:
	@echo "--- Deploying Tinker RL Basic Job to GKE ---"
	kubectl delete job tinker-rl-basic-job --ignore-not-found=true
	kubectl apply -f client/kubernetes/tinker-rl-basic-job.yaml
	@echo "Waiting for job to start..."
	@sleep 4
	kubectl logs -f job/tinker-rl-basic-job

stop-tinker-job:
	@echo "--- Stopping Tinker RL Basic Job ---"
	kubectl delete job tinker-rl-basic-job --ignore-not-found=true

logs-tinker-job:
	@echo "--- Fetching Tinker RL Basic Job Logs ---"
	kubectl logs -f job/tinker-rl-basic-job

run-tinker-job-2:
	@echo "--- Deploying Tinker RL Basic Job 2 to GKE ---"
	kubectl delete job tinker-rl-basic-job-2 --ignore-not-found=true
	kubectl apply -f client/kubernetes/tinker-rl-basic-job-2.yaml
	@echo "Waiting for job to start..."
	@sleep 4
	kubectl logs -f job/tinker-rl-basic-job-2

stop-tinker-job-2:
	@echo "--- Stopping Tinker RL Basic Job 2 ---"
	kubectl delete job tinker-rl-basic-job-2 --ignore-not-found=true

logs-tinker-job-2:
	@echo "--- Fetching Tinker RL Basic Job 2 Logs ---"
	kubectl logs -f job/tinker-rl-basic-job-2

run-tinker-job-3:
	@echo "--- Deploying Tinker RL Basic Job 3 to GKE ---"
	kubectl delete job tinker-rl-basic-job-3 --ignore-not-found=true
	kubectl apply -f client/kubernetes/tinker-rl-basic-job-3.yaml
	@echo "Waiting for job to start..."
	@sleep 4
	kubectl logs -f job/tinker-rl-basic-job-3

stop-tinker-job-3:
	@echo "--- Stopping Tinker RL Basic Job 3 ---"
	kubectl delete job tinker-rl-basic-job-3 --ignore-not-found=true

logs-tinker-job-3:
	@echo "--- Fetching Tinker RL Basic Job 3 Logs ---"
	kubectl logs -f job/tinker-rl-basic-job-3

run-client-job-parallel:
	@echo "--- Deploying Distributed RLVR Client Job Array to GKE ---"
	kubectl delete job open-rl-client-job-parallel --ignore-not-found=true
	kubectl apply -f client/kubernetes/rlvr-job-parallel.yaml
	@echo "Waiting for jobs to start..."
	@sleep 6
	@echo "Tailing one of the array pods..."
	kubectl logs -f job/open-rl-client-job-parallel

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

# --- Cloud Monitoring Dashboard ---
dashboard-apply:
	@scripts/apply_dashboard.sh $(GCP_PROJECT)
