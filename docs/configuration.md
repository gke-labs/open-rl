# Configuration

All knobs that control Open-RL at runtime. Read from environment variables.
The Makefile wraps the common ones; k8s manifests set them in pod specs.

## Core

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | *(required in single-process mode)* | The Hugging Face model id to load in the trainer **and** the vLLM sampler. Example: `google/gemma-4-e2b`. |
| `SAMPLER` | `torch` when single-process, `vllm` otherwise | Which sampling backend to use. `torch` runs in the gateway process (CPU-friendly). `vllm` forwards sampling requests to a separate `python -m src.vllm_sampler` worker. |
| `SINGLE_PROCESS` | auto-detected | `1` forces gateway + trainer into one process (the sampler is separate iff `SAMPLER=vllm`). Unset falls back to auto-detect: if `BASE_MODEL` is set and `REDIS_URL` is not, assume single-process. Distributed deployments shouldn't set this. |
| `REDIS_URL` | unset | When set, the request store switches to Redis and the system runs in distributed mode (gateway, vLLM worker, and trainer worker as separate pods coordinating via Redis). Used by the k8s manifests in `server/kubernetes/distributed-{shared,lustre}/`. |
| `VLLM_URL` | `http://127.0.0.1:8001` | Where the gateway looks for the vLLM worker when `SAMPLER=vllm`. The `make vllm` target starts the worker on this port. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root dir for adapter snapshots (`peft/`) and saved state checkpoints (`checkpoints/`). |
| `CUDA_VISIBLE_DEVICES` | unset | Standard PyTorch GPU selector. Pinned per-deployment in the k8s manifests. |

## vLLM worker tuning

| Env var | Default | What it does |
| --- | --- | --- |
| `MOCK_VLLM` | `0` | `1` stubs out vLLM so the worker returns dummy tokens. Useful on a Mac where vLLM isn't installable. |

## Client

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` / `OPEN_RL_BASE_URL` | `http://127.0.0.1:9003` | Where the client scripts point the tinker SDK. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the SDK. No real auth on the local server; any value works. |
| `HF_TOKEN` | unset | Required for gated models (Gemma 3, FunctionGemma). `uv run hf auth login` is the easiest way to set it. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout (debugging). |

## Quick reference

```bash
# Local single-process, torch sampler (default):
make server BASE_MODEL=Qwen/Qwen3-0.6B

# Local single-process, vLLM sampler (GPU):
make vllm   BASE_MODEL=google/gemma-4-e2b   # terminal 1
make server BASE_MODEL=google/gemma-4-e2b SAMPLER=vllm   # terminal 2

# Distributed (GKE): configured via k8s manifests, no local command.
```
