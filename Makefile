.PHONY: run-server run-server-engine-sampler run-function-gemma-server run-function-gemma-sft run-pig-latin-server run-pig-latin-sft run-text-to-sql-server run-text-to-sql-server-gpu run-text-to-sql-vllm run-text-to-sql-sft run-sft run-sft-parallel run-rlvr run-rlvr-parallel test lint fmt

# Default VLLM model for inference, can be overridden via `make run-vllm VLLM_MODEL=...`
#VLLM_MODEL ?= Qwen/Qwen2.5-0.5B
VLLM_MODEL ?= Qwen/Qwen3-4B-Instruct-2507

# Default GPU allocations for running isolated processes locally/on VMs
TRAINER_GPU ?= 0
VLLM_GPU ?= 1

# Run the gateway-only server locally
run-server:
	@-kill -9 $$(lsof -ti:8000) 2>/dev/null || true
	cd server && UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(TRAINER_GPU)" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Kill any local process stuck listening on port 8000
kill-server:
	@-kill -9 $$(lsof -ti:8000) 2>/dev/null || echo "Port 8000 is free"

# Run the standalone vLLM inference worker locally
run-vllm:
	cd server && UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(VLLM_GPU)" VLLM_MODEL="$(VLLM_MODEL)" uv run --extra gpu --extra vllm python -m src.vllm_worker

run-server-engine-sampler:
	@-kill -9 $$(lsof -ti:8000) 2>/dev/null || true
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" SAMPLER_BACKEND=engine VLLM_MODEL="$(VLLM_MODEL)" uv run --extra cpu uvicorn src.main:app --host 127.0.0.1 --port 8000

run-function-gemma-server:
	@-kill -9 $$(lsof -ti:9000) 2>/dev/null || true
	cd server && OPEN_RL_SINGLE_PROCESS=1 SAMPLER_BACKEND=engine OPEN_RL_BASE_MODEL="google/functiongemma-270m-it" PYTHONUNBUFFERED=1 uv run --extra cpu uvicorn src.main:app --host 127.0.0.1 --port 9000 $(ARGS)

run-function-gemma:
	cd client && uv run --python 3.12 functiongemma-demo $(ARGS)

run-pig-latin-server:
	@-kill -9 $$(lsof -ti:9001) 2>/dev/null || true
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" OPEN_RL_SINGLE_PROCESS=1 OPEN_RL_BASE_MODEL="Qwen/Qwen3-0.6B" SAMPLER_BACKEND=engine VLLM_MODEL="Qwen/Qwen3-0.6B" uv run --extra cpu uvicorn src.main:app --host 127.0.0.1 --port 9001

run-pig-latin-sft:
	cd client && uv run --python 3.12 -i https://pypi.org/simple python -u piglatin_sft.py qwen $(ARGS)

run-pig-latin-gemma-server:
	@-kill -9 $$(lsof -ti:9002) 2>/dev/null || true
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" OPEN_RL_SINGLE_PROCESS=1 OPEN_RL_BASE_MODEL="google/gemma-3-1b-it" SAMPLER_BACKEND=engine VLLM_MODEL="google/gemma-3-1b-it" uv run --extra cpu uvicorn src.main:app --host 127.0.0.1 --port 9002

run-pig-latin-gemma-sft:
	cd client && uv run --python 3.12 -i https://pypi.org/simple python -u piglatin_sft.py gemma base_url="http://127.0.0.1:9002" $(ARGS)

TEXT_TO_SQL_SERVER_EXTRA ?= cpu
TEXT_TO_SQL_BASE_MODEL ?= google/gemma-3-1b-pt
TEXT_TO_SQL_SAMPLER_BACKEND ?= engine
TEXT_TO_SQL_VLLM_URL ?= http://127.0.0.1:8001

run-text-to-sql-server:
	@-kill -9 $$(lsof -ti:9003) 2>/dev/null || true
	cd server && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(TRAINER_GPU)" OPEN_RL_SINGLE_PROCESS=1 OPEN_RL_BASE_MODEL="$(TEXT_TO_SQL_BASE_MODEL)" SAMPLER_BACKEND="$(TEXT_TO_SQL_SAMPLER_BACKEND)" VLLM_MODEL="$(TEXT_TO_SQL_BASE_MODEL)" VLLM_URL="$(TEXT_TO_SQL_VLLM_URL)" uv run --extra $(TEXT_TO_SQL_SERVER_EXTRA) uvicorn src.main:app --host 127.0.0.1 --port 9003

run-text-to-sql-server-gpu:
	$(MAKE) run-text-to-sql-server TEXT_TO_SQL_SERVER_EXTRA=gpu TEXT_TO_SQL_SAMPLER_BACKEND=vllm

run-text-to-sql-vllm:
	cd server && UV_INDEX_URL="https://pypi.org/simple" CUDA_VISIBLE_DEVICES="$(VLLM_GPU)" VLLM_MODEL="$(TEXT_TO_SQL_BASE_MODEL)" uv run --extra gpu --extra vllm python -m src.vllm_worker

TEXT_TO_SQL_PRESET ?= gemma

run-text-to-sql-sft:
	cd client && uv run --python 3.12 -i https://pypi.org/simple python -u texttosql_sft.py $(TEXT_TO_SQL_PRESET) base_url="http://127.0.0.1:9003" $(ARGS)

test:
	cd client && uv run python -m unittest discover tests

lint:
	uvx ruff check .
	uvx ruff format --check .

fmt:
	uvx ruff check --fix .
	uvx ruff format .

# Client test targets
run-sft:
	cd client && uv run -i https://pypi.org/simple python sft.py --base-model "$(VLLM_MODEL)" $(ARGS)

run-sft-parallel:
	cd client && uv run -i https://pypi.org/simple python sft.py --parallel --base-model "$(VLLM_MODEL)"

# Default concurrent jobs for parallel execution
JOBS ?= 2
STEPS ?= 15

# OpenTelemetry Tracing Toggles (0 = disabled, 1 = enabled)
ENABLE_GCP_TRACE ?= 0
ENABLE_CONSOLE_TRACE ?= 0

run-rlvr:
	cd client && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) ENABLE_CONSOLE_TRACE=$(ENABLE_CONSOLE_TRACE) uv run -i https://pypi.org/simple python rlvr.py --jobs 1 --steps $(STEPS) --base-model "$(VLLM_MODEL)"

run-rlvr-parallel:
	cd client && ENABLE_GCP_TRACE=$(ENABLE_GCP_TRACE) ENABLE_CONSOLE_TRACE=$(ENABLE_CONSOLE_TRACE) uv run -i https://pypi.org/simple python rlvr.py --jobs $(JOBS) --steps $(STEPS) --base-model "$(VLLM_MODEL)"

# Plot metrics from a JSONL file
# Usage: make plot-metrics [FILE=path/to/metrics.jsonl]
plot-metrics:
	cd client && uv run -i https://pypi.org/simple python plot_metrics.py $(FILE)

# Plot parallel metrics from the RLVR log file
# Usage: make plot-logs [LOG_FILE=client/rlvr_parallel_results.log] [WATCH=1]
plot-logs:
	cd client && uv run -i https://pypi.org/simple python plot_logs.py $(or $(LOG_FILE),rlvr_parallel_results.log) $(if $(WATCH),--watch,)

# Generate diagrams using local mmdc zsh alias
diagrams:
	zsh -ic "mmdc -i assets/architecture.mmd -o assets/architecture.svg -b transparent"
	zsh -ic "mmdc -i assets/architecture.mmd -o assets/architecture.png -s 3 -b transparent"

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
	@cd client && uv run -i https://pypi.org/simple python cli.py $(CLI_ARGS)

# Shortcut: make run-cli-list
run-cli-list:
	@cd client && uv run -i https://pypi.org/simple python cli.py list

# Shortcut: make run-cli-chat MODEL=... [PROMPT="..."]
run-cli-chat:
	@test -n "$(MODEL)" || (echo "Error: MODEL argument is required. Usage: make run-cli-chat MODEL=<model_id>" && exit 1)
	@cd client && uv run -i https://pypi.org/simple python cli.py chat --model $(MODEL) --system-prompt "$(or $(PROMPT),You are helpful geography assistant.)"

# --- Deployment Targets ---

GCP_PROJECT ?= cdrollouts-sunilarora
GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-server
GATEWAY_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-gateway
CLIENT_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/open-rl-client
TINKER_GCR_REPO ?= gcr.io/$(GCP_PROJECT)/tinker-cookbook
IMAGE_TAG ?= latest

build-server-images:
	@echo "--- Building Server Docker Images ---"
	cd server && DOCKER_BUILDKIT=1 docker build -t $(GCR_REPO):$(IMAGE_TAG) -f Dockerfile .
	cd server && DOCKER_BUILDKIT=1 docker build -t $(GATEWAY_GCR_REPO):$(IMAGE_TAG) -f Dockerfile.gateway .

push-server-images:
	@echo "--- Pushing Server Images to GCR ---"
	docker push $(GCR_REPO):$(IMAGE_TAG)
	docker push $(GATEWAY_GCR_REPO):$(IMAGE_TAG)

deploy-server:
	@echo "-- Deploy OpenRL Stack to Kubernetes Cluster ---"
	kubectl apply -k server/kubernetes/distributed-lustre/

rollout:
	@echo "--- Rolling out latest server deployments ---"
	kubectl rollout restart deployment redis-broker open-rl-gateway open-rl-trainer-worker vllm-worker
	kubectl rollout status deployment open-rl-gateway
	kubectl rollout status deployment open-rl-trainer-worker

build-client-image:
	@echo "--- Building Client Docker Image ---"
	cd client && DOCKER_BUILDKIT=1 docker build -t $(CLIENT_GCR_REPO):$(IMAGE_TAG) .

push-client-image:
	@echo "--- Pushing Client Image to GCR ---"
	docker push $(CLIENT_GCR_REPO):$(IMAGE_TAG)

build-tinker-image:
	@echo "--- Building tinker-cookbook Docker Image ---"
	cd ../tinker-cookbook && DOCKER_BUILDKIT=1 docker build -t $(TINKER_GCR_REPO):$(IMAGE_TAG) .

push-tinker-image:
	@echo "--- Pushing tinker-cookbook Image to GCR ---"
	docker push $(TINKER_GCR_REPO):$(IMAGE_TAG)

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
