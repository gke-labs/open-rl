FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/autoresearch \
    GEMINI_CLI_TRUST_WORKSPACE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    nodejs \
    npm \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

RUN npm install -g @google/gemini-cli

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY autoresearch ./autoresearch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

WORKDIR /app/autoresearch

CMD ["uv", "run", "--no-sync", "python", "-m", "harness.agent"]
