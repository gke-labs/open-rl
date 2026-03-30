# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# Install Python 3.12 via uv
RUN uv python install 3.12

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Configure uv for Docker and caching
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies using uv with BuildKit cache
# --frozen ensures it uses the uv.lock exactly
# --all-extras ensures the optional train/gpu stacks are installed for the worker image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra gpu --extra vllm

# Copy source code
COPY src ./src

# Expose ports (Gateway: 8000, vLLM Worker: 8001)
EXPOSE 8000
EXPOSE 8001

# Start the Gateway using uv run to ensure the venv is active
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
