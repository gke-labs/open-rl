# Server Setup

Use the client README for the full repo bootstrap flow:

- [Client README](../client/README.md)

Then pick the server environment that matches your local workflow:

- local single-process (CPU, torch sampler):
  `cd server && uv sync --extra cpu && SINGLE_PROCESS=1 SAMPLER=torch BASE_MODEL="Qwen/Qwen3-0.6B" uv run --extra cpu uvicorn src.gateway:app --host 127.0.0.1 --port 9003`
- standalone vLLM worker (GPU):
  `cd server && uv sync --extra gpu --extra vllm && BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" uv run --extra gpu --extra vllm python -m src.vllm_sampler`
