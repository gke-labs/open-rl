.PHONY: run-server run-sft run-sft-parallel run-rlvr run-rlvr-parallel

# Run the Uvicorn server locally, forcing the public PyPI index for uv
run-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Client test targets
run-sft:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_sft.py $(ARGS)

run-sft-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_sft.py --parallel

run-rlvr:
	cd client && uv run --no-sync -i https://pypi.org/simple python showcase_rlvr.py

run-rlvr-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python showcase_rlvr.py parallel

# Generate diagrams using local mmdc zsh alias
diagrams:
	zsh -ic "mmdc -i design_arch.mmd -o design_arch.svg"
	zsh -ic "mmdc -i rollout_flow.mmd -o rollout_flow.svg"

# Sync server to remote host b1
# TODO: sync only server directory
# TODO: avoid syncing pycache files
server-sync:
	rsync -avz --exclude '.git' --exclude '.venv' ./ b1:~/work/kube-rl

server-tunnel:
	ssh -fN -L 8000:localhost:8000 b1

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