# Use NVIDIA CUDA 12.9.0 devel base image for PyTorch/vLLM runtime compilation
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_FROZEN=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# Install Python 3.12 via uv
RUN uv python install 3.12

# Install the NVIDIA CUDA checkpoint/restore utility used by the in-repo
# snapshot agent. The utility is driver-backed; keep the node driver compatible.
RUN curl -fsSL https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint \
    -o /usr/local/bin/cuda-checkpoint \
    && chmod +x /usr/local/bin/cuda-checkpoint

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Configure uv for Docker and caching
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies using uv with BuildKit cache
# --frozen ensures it uses the uv.lock exactly
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra gpu --extra vllm --extra cluster

# Copy source code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra gpu --extra vllm --extra cluster

# Apply vLLM patches (Gemma 4 LoRA dedup)
RUN uv run python /app/src/server/scripts/patch_vllm_lora_dedup.py


ENV PATH="/usr/local/cuda/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Expose ports (Gateway: 8000, vLLM Worker: 8001)
EXPOSE 8000
EXPOSE 8001

# Start the Gateway using uv run to ensure the venv is active
CMD ["uv", "run", "uvicorn", "server.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
