# NVIDIA CUDA 12.9.0 *base* image (driver shim + cudart only, ~0.45 GB).
#
# The devel image this used to build on is ~10 GB, almost all of it a second
# copy of the CUDA libraries: torch and vLLM link against the pip-installed
# nvidia-*-cu12 wheels inside /app/.venv, not /usr/local/cuda. The only pieces
# of the toolkit the image needs from outside the venv are the ones the runtime
# JIT compilers (FlashInfer, tilelang, torch cpp extensions) shell out to: the
# 12.9 `nvcc` and the cudart/driver headers and stubs. Those are installed
# below from the CUDA apt repo the base image already configures, so that
# CUDA_HOME=/usr/local/cuda stays a complete, version-matched toolchain root.
# (The pip nvidia-cuda-nvcc-cu12 wheel ships only ptxas; the only nvcc binary
# in the venv is the CUDA 13.2 one under nvidia/cu13, which cannot drive
# 12.9 builds.)
FROM nvidia/cuda:12.9.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_FROZEN=1

# Install dependencies. python3-dev is required for compiling Triton runtime
# kernels during vLLM engine initialization. cuda-nvcc-12-9 (nvcc, cicc,
# ptxas, nvvm) and cuda-cudart-dev-12-9 (cuda.h, cuda_runtime.h, cuda_bf16.h,
# CCCL headers, libcudart.so and the libcuda.so link stub) are the runtime-JIT
# subset of the devel image; versions are pinned like the upstream images do.
ARG NV_CUDA_NVCC_VERSION=12.9.86-1
ARG NV_CUDA_CUDART_DEV_VERSION=12.9.79-1
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3-dev \
    ninja-build \
    cuda-nvcc-12-9=${NV_CUDA_NVCC_VERSION} \
    cuda-cudart-dev-12-9=${NV_CUDA_CUDART_DEV_VERSION} \
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

# Configure uv for Docker and caching.
#
# UV_COMPILE_BYTECODE: without it the venv ships 24k .py files and no .pyc, so
# every worker start recompiles the whole import graph and throws it away --
# `import vllm` measured 12.9s cold against 5.6s with bytecode present, on every
# trainer and sampler launch, for ~0.4GB of image. The API server and client images
# already set this.
#
# UV_NO_SYNC: the source is exposed through PYTHONPATH below rather than
# installed, so the implicit sync `uv run` performs at container start only
# rebuilds and reinstalls the project to no effect.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1

# Install dependencies using uv with BuildKit cache (fast dependency extraction)
# --frozen ensures it uses the uv.lock exactly
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra gpu --extra vllm --extra cluster

# Expose the pip CUDA wheels' headers and link-time library names through
# CUDA_HOME. The devel image supplied these via the lib*-dev debs; runtime JIT
# needs them (FlashInfer's sampling kernels include curand.h, its GEMM modules
# cublas_v2.h, fused MoE links -lnvrtc; tilelang also reaches for cudnn.h and
# nvshmem.h). Symlinks only, so the libraries are not duplicated and existing
# apt-provided files (cudart, driver, CCCL, crt) win. nvidia/cu13 is skipped:
# it is the CUDA 13 toolkit cuda-tile drags in and must not shadow 12.9.
# Both the SONAME (lib*.so.N) and the unversioned link name are mirrored, as
# in the devel image: ld resolves DT_NEEDED entries such as libcublas ->
# libcublasLt.so.12 by the versioned name. At runtime the symlinks resolve to
# the very files torch already loads, and glibc dedups by inode.
RUN set -eu; \
    sp=/app/.venv/lib/python3.12/site-packages/nvidia; \
    inc=/usr/local/cuda/include; lib=/usr/local/cuda/lib64; \
    for d in "$sp"/*/; do \
        case "$d" in */cu13/) continue;; esac; \
        if [ -d "$d/include" ]; then cp -rsn "$d/include/." "$inc/"; fi; \
        [ -d "$d/lib" ] || continue; \
        for so in "$d"lib/lib*.so.*; do \
            [ -e "$so" ] || continue; \
            name=$(basename "$so"); base="${name%%.so.*}.so"; \
            [ -e "$lib/$name" ] || ln -s "$so" "$lib/$name"; \
            [ -e "$lib/$base" ] || ln -s "$so" "$lib/$base"; \
        done; \
    done

# Copy source code
COPY . .

# Install the project, including the vLLM general plugin entry point used by
# spawned workers to discover the checkpoint weight-transfer backend.
RUN uv sync --frozen --extra gpu --extra vllm --extra cluster

# The venv is on PATH so a job that names this image as its trainer runs
# `python -u -m server.training_requests_processor` like any other image.
ENV PATH="/app/.venv/bin:/usr/local/cuda/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    # The devel image exported this so JIT-built extensions can link -lcuda
    # against the driver stub; the base image does not.
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \
    PYTHONPATH="/app/src:/app:$PYTHONPATH" \
    # GKE's device plugin mounts the NVIDIA driver (libcuda.so) at
    # /usr/local/nvidia; the 12.9.0-devel base image dropped it from
    # LD_LIBRARY_PATH, silently leaving torch/vLLM CPU-only on GKE.
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

# Expose ports (API server: 8000, vLLM Worker: 8001)
EXPOSE 8000
EXPOSE 8001

# Start the API server using uv run to ensure the venv is active
CMD ["uv", "run", "uvicorn", "server.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
