# Open-RL Client

This directory contains the client-side scripts and SDK for interacting with the Open-RL API.

## Getting Started with `uv`

This repo uses `uv` for both the client and server. From a fresh machine:

### 1. Install `uv`

On macOS:

```bash
brew install uv
```

Or on macOS/Linux with the upstream installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify:

```bash
uv --version
```

### 2. Clone the repo

```bash
git clone <your-open-rl-repo-url>
cd open-rl
```

### 3. Sync the two Python projects

The repo is split into:

- `server/` for the gateway, trainer worker, and sampler worker
- `client/` for demos and training scripts

Sync the client:

```bash
cd client
uv sync
cd ..
```

Then choose the server environment you need:

Gateway/core only:

```bash
cd server
uv sync
cd ..
```

Local single-process training flows such as Pig Latin SFT or FunctionGemma (CPU PyTorch):

```bash
cd server
uv sync --extra cpu
cd ..
```

Linux GPU/vLLM worker flows (CUDA PyTorch on Linux/WSL):

```bash
cd server
uv sync --extra gpu --extra vllm
cd ..
```

### 4. Run common workflows with `uv`

Start the server (or use `make server`):

```bash
cd server
SINGLE_PROCESS=1 \
BASE_MODEL="Qwen/Qwen3-0.6B" \
SAMPLER=torch \
uv run --extra cpu uvicorn src.gateway:app --host 127.0.0.1 --port 9003
```

Start a standalone vLLM worker (or use `make vllm`):

```bash
cd server
BASE_MODEL="Qwen/Qwen3-0.6B" \
uv run --extra vllm python -m src.vllm_sampler
```

Run the Pig Latin SFT example:

```bash
cd client
uv run python -u piglatin_sft.py qwen base_url="http://127.0.0.1:9001"
```

Run the RLVR demo:

```bash
cd client
TINKER_BASE_URL="http://127.0.0.1:8000" \
uv run python rlvr.py --jobs 1 --steps 5 --base-model "Qwen/Qwen3-4B-Instruct-2507"
```

You can also use the repo Make targets if you prefer:

```bash
make run-server
make run-pig-latin-server
make run-pig-latin-sft
make run-rlvr
```

Notes:

- `server/uv sync --extra cpu` installs the local training stack with CPU PyTorch for the single-process engine flows.
- `server/uv sync --extra gpu --extra vllm` adds the Linux-only vLLM worker dependencies and resolves CUDA PyTorch wheels.
- `vllm` is Linux-only here. On a Mac, use the gateway-only or single-process `cpu` flows unless you are running the Linux container story.
- `tinker-cookbook` is not required for the standard client demos in this repo.
- FunctionGemma examples require Hugging Face auth and model access.

## Available Guides & Examples

Detailed walkthroughs for building with the Open-RL framework have been moved to the centralized docs folder:

- **Supervised Fine-Tuning**
  - [FunctionGemma Demo](../docs/guides/supervised/function-gemma.md)
  - [Pig Latin SFT](../docs/guides/supervised/pig-latin.md)
- **Reinforcement Learning**
  - [RLVR Demo](../docs/guides/reinforcement-learning/rlvr.md)
