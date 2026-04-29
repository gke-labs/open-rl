# Configuration

Open-RL is configured with environment variables. The examples below use plain
shell commands so they work even if `make` is not installed. The root
`Makefile` wraps the same commands for convenience.

## Run locally

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Start the gateway and trainer with the default torch sampler:

```bash
cd src/server
BASE_MODEL=Qwen/Qwen3-0.6B \
SAMPLER=torch \
uv run --extra cpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

Local mode is auto-detected when `BASE_MODEL` is set and `REDIS_URL` is unset.
No separate single-process setting is required.

For a separate vLLM sampler, use two terminals:

```bash
# Terminal 1: vLLM sampler
cd src/server
BASE_MODEL=google/gemma-4-e2b \
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
CUDA_VISIBLE_DEVICES=0 \
uv run --extra vllm python -m vllm_sampler
```

```bash
# Terminal 2: gateway and trainer
cd src/server
BASE_MODEL=google/gemma-4-e2b \
SAMPLER=vllm \
CUDA_VISIBLE_DEVICES=1 \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

The equivalent Makefile shortcuts are:

```bash
make server BASE_MODEL=Qwen/Qwen3-0.6B
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM make vllm BASE_MODEL=google/gemma-4-e2b
make server BASE_MODEL=google/gemma-4-e2b SAMPLER=vllm
```

## Core variables

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | unset | Hugging Face model id loaded by the trainer and, when using vLLM, by the sampler. |
| `SAMPLER` | `torch` locally, `vllm` when distributed | Sampling backend. `torch` samples in the training process. `vllm` forwards sampling requests to a vLLM worker. |
| `REDIS_URL` | unset | Enables distributed mode by switching the request store to Redis. Leave unset for local mode. |
| `VLLM_URL` | `http://127.0.0.1:8001` | Gateway URL for the vLLM worker when `SAMPLER=vllm`. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root directory for adapter snapshots under `peft/` and saved states under `checkpoints/`. |
| `CUDA_VISIBLE_DEVICES` | unset | Standard PyTorch GPU selector. Use different devices when the vLLM worker and trainer run on separate GPUs. |

## vLLM variables

| Env var | Default | What it does |
| --- | --- | --- |
| `MOCK_VLLM` | `0` | `1` starts the vLLM worker without a real vLLM engine, useful for local API debugging. |
| `VLLM_ARCHITECTURE_OVERRIDE` | unset | Optional architecture override passed to the in-repo vLLM worker. Gemma 4 examples use `Gemma4ForCausalLM`. |

## Client variables

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` | `http://127.0.0.1:9003` | Base URL used by example clients and scripts. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the Tinker SDK. Local Open-RL does not enforce auth. |
| `HF_TOKEN` | unset | Required for gated Hugging Face models. `uv run hf auth login` is the easiest setup path. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout for debugging. |

## Distributed deploys

Kubernetes deploys set these variables in pod specs. The important split is:

```bash
# Gateway pod
REDIS_URL=redis://redis-service:6379 \
VLLM_URL=http://vllm-service:8001 \
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
uv run uvicorn src.gateway:app --host 0.0.0.0 --port 8000
```

```bash
# Trainer worker pod
REDIS_URL=redis://redis-service:6379 \
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
uv run python -m src.clock_cycle
```

```bash
# vLLM worker pod
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
uv run uvicorn src.vllm_sampler:app --host 0.0.0.0 --port 8001
```
