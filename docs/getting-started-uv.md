# Getting Started with `uv`

This guide covers the fastest path from a fresh machine to a working Open-RL checkout using `uv`.

## 1. Install `uv`

Pick one of the supported install paths.

### macOS with Homebrew

```bash
brew install uv
```

### macOS or Linux with the upstream installer

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, then verify:

```bash
uv --version
```

## 2. Clone the repo

```bash
git clone <your-open-rl-repo-url>
cd open-rl
```

## 3. Install Python with `uv`

Both the client and server default to Python 3.12.8 through their local `.python-version` files, so install that interpreter first:

```bash
uv python install 3.12.8
```

You can confirm what `uv` will pick up:

```bash
cd server && uv python find
cd ../client && uv python find
cd ..
```

## 4. Sync dependencies

The repo is split into two Python projects:

- `server/` for the gateway, trainer worker, and vLLM worker
- `client/` for demos and training scripts

Sync them separately.

### Server

For the full Linux/GPU server environment:

```bash
cd server
uv sync --extra ml
cd ..
```

For a lighter gateway-only environment:

```bash
cd server
uv sync
cd ..
```

### Client

```bash
cd client
uv sync
cd ..
```

## 5. Run the most common commands with `uv`

### Start the gateway locally

```bash
cd server
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

### Start the local single-process Pig Latin server

```bash
cd server
OPEN_RL_SINGLE_PROCESS=1 \
OPEN_RL_BASE_MODEL="Qwen/Qwen3-0.6B" \
SAMPLER_BACKEND=engine \
VLLM_MODEL="Qwen/Qwen3-0.6B" \
uv run uvicorn src.main:app --host 127.0.0.1 --port 9001
```

### Run the Pig Latin SFT example

In another terminal:

```bash
cd client
uv run --python 3.12 --no-sync python -u piglatin_sft.py qwen base_url="http://127.0.0.1:9001"
```

### Run the RLVR demo

```bash
cd client
TINKER_BASE_URL="http://127.0.0.1:8000" \
uv run --python 3.12 --no-sync python rlvr.py --jobs 1 --steps 5 --base-model "Qwen/Qwen3-4B-Instruct-2507"
```

## 6. Use the repo Make targets if you prefer

The Makefile already wraps the main `uv run` commands:

```bash
make run-server
make run-pig-latin-server
make run-pig-latin-sft
make run-rlvr
```

Those targets still rely on `uv` underneath, so you only need the sync steps above.

## 7. Fresh-system notes

- `server/uv sync --extra ml` is the expensive step because it pulls PyTorch, Transformers, PEFT, and optionally vLLM.
- `vllm` is Linux-only in this repo. On a Mac, use the gateway-only flow or the Docker smoke-test flow instead of expecting local CUDA inference.
- `tinker-cookbook` is not required for the standard server or client demos in this repo.
- FunctionGemma examples require Hugging Face auth and model access.

## 8. Useful `uv` maintenance commands

Upgrade lockfile-driven environments after dependency changes:

```bash
cd server && uv sync --extra ml
cd client && uv sync
```

See interpreter and environment details:

```bash
uv python list
cd server && uv tree
cd client && uv tree
```
