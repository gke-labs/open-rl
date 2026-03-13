# Server Setup

Use the repo-level `uv` setup guide first:

- [Getting Started with `uv`](../docs/getting-started-uv.md)

Then pick one of these server flows:

- gateway only: `cd server && uv sync && uv run uvicorn src.main:app --host 127.0.0.1 --port 8000`
- full ML environment: `cd server && uv sync --extra ml`
