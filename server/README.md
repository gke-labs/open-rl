# Server Setup

Use the client README for the full repo bootstrap flow:

- [Client README](../client/README.md)

Then pick the server environment that matches your local workflow:

- gateway/core only:
  `cd server && uv sync && uv run uvicorn src.main:app --host 127.0.0.1 --port 8000`
- local single-process training:
  `cd server && uv sync --extra train && OPEN_RL_SINGLE_PROCESS=1 SAMPLER_BACKEND=engine OPEN_RL_BASE_MODEL="Qwen/Qwen3-0.6B" uv run --extra train uvicorn src.main:app --host 127.0.0.1 --port 9001`
- Linux GPU/vLLM worker:
  `cd server && uv sync --extra gpu && CUDA_VISIBLE_DEVICES=0 VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507" uv run --extra gpu python -m src.vllm_worker`
