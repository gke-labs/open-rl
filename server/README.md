# Server Setup

Use the client README for the `uv` bootstrap flow:

- [Client README](../client/README.md)

Then pick one of these server flows:

- gateway only: `cd server && uv sync && uv run uvicorn src.main:app --host 127.0.0.1 --port 8000`
- full ML environment: `cd server && uv sync --extra ml`
