#!/usr/bin/env bash
# Build the interpreter the Automodel trainer runs on.
#
# Why this is not a uv extra in pyproject.toml, same as the Megatron env:
# nemo-automodel pins transformers exactly and pulls megatron-fsdp, torchao,
# quack-kernels and mlflow, none of which the samplers or the FSDP path want
# resolved into .venv. It gets its own venv, plus the open-rl trainer's own
# dependencies, because the trainer runs server.training_requests_processor
# off PYTHONPATH rather than an install.
#
# Two things a forward pass will not tell you (see the qwen35 GDN notes):
# the GDN backward on Hopper needs tilelang or fla refuses it, and tilelang
# 0.1.9 needs apache-tvm-ffi pinned to 0.1.9. causal-conv1d compiles against
# this env's torch, which is why it is installed without build isolation.
#
# Verified on the 8xH200 box (CUDA 12.9, driver cu129) with nemo-automodel
# 0.6.0 / torch 2.11.0+cu129 / transformers 5.12.1 / fla 0.5.1 / tilelang 0.1.9.
#
#   ./scripts/setup_automodel_env.sh              # builds ~/automodel/.venv
#   AUTOMODEL_VENV=~/other ./scripts/setup_automodel_env.sh
#
# Point launch_work.sh at the result with AUTOMODEL_PYTHON=<venv>/bin/python.
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VENV="${AUTOMODEL_VENV:-$HOME/automodel/.venv}"
V="$VENV/bin/python"
TORCH_INDEX=https://download.pytorch.org/whl/cu129

if [ ! -x "$V" ]; then
  uv venv --python 3.12 "$VENV"
fi

uv pip install --python "$V" torch==2.11.0 torchvision --index-url "$TORCH_INDEX"

# The open-rl trainer's runtime imports.
uv pip install --python "$V" \
  packaging ninja pybind11 einops peft accelerate safetensors rich regex pyyaml tqdm omegaconf \
  "transformers==5.12.1" datasets "chz>=0.4.0" fastapi pydantic "redis>=5.0.0" uvicorn httpx psutil setuptools \
  opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-gcp-trace

# nemo-automodel with its dependencies, torch held at the cu129 build.
uv pip install --python "$V" "nemo-automodel==0.6.0" "torch==2.11.0" "transformers==5.12.1" \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# GDN kernels: fla for the delta rule (its ops.cp is what Automodel's GDN
# context parallelism runs on), tilelang for the Hopper backward.
uv pip install --python "$V" "flash-linear-attention==0.5.1" "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "torch==2.11.0" \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
MAX_JOBS=$(nproc) uv pip install --python "$V" --no-build-isolation "causal-conv1d>=1.4"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$REPO/src" "$V" - <<'EOF'
import torch, transformers
from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM  # noqa: F401
import fla.ops.cp  # noqa: F401
import tilelang  # noqa: F401
import causal_conv1d  # noqa: F401
import server.training_requests_processor  # noqa: F401
import training.automodel_worker  # noqa: F401

print(f"automodel env OK: torch {torch.__version__}, transformers {transformers.__version__}")
EOF
echo "[setup] done: $V"
