#!/usr/bin/env bash
# Build the interpreter the Automodel trainer runs on.
#
# Why this is not a uv extra in pyproject.toml:
# nemo-automodel pins transformers exactly and pulls megatron-fsdp, torchao,
# quack-kernels and mlflow, none of which the samplers or the FSDP path want
# resolved into .venv. It gets its own venv, plus the open-rl trainer's own
# dependencies, because the trainer runs server.training_requests_processor
# off PYTHONPATH rather than an install.
#
# Two things a forward pass will not tell you (see the qwen35 GDN notes):
# the GDN backward on Hopper needs tilelang or fla refuses it, and tilelang
# 0.1.9 needs apache-tvm-ffi pinned to 0.1.9.
#
# The 0.6.0 release was verified on the 8xH200 box (CUDA 12.9, driver cu129)
# with torch 2.11.0+cu129 / transformers 5.12.1 / fla 0.5.1 / tilelang 0.1.9.
# The pin below is a main commit past 0.6.0 for its Datum/collate_datums and
# merged PEFT export; it moves transformers to 5.15.1 and tilelang to 0.1.11,
# and is not verified on the box yet.
#
#   ./scripts/setup_automodel_env.sh              # builds ~/automodel/.venv
#   AUTOMODEL_VENV=~/other ./scripts/setup_automodel_env.sh
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VENV="${AUTOMODEL_VENV:-$HOME/automodel/.venv}"
V="$VENV/bin/python"
TORCH_INDEX=https://download.pytorch.org/whl/cu129
AUTOMODEL_SHA="${AUTOMODEL_SHA:-5811fc844e4d8ba112b60fc7b3c05f30532bbc15}"
TRANSFORMERS="transformers==5.15.1"

if [ ! -x "$V" ]; then
  uv venv --python 3.12 "$VENV"
fi

uv pip install --python "$V" torch==2.11.0 torchvision --index-url "$TORCH_INDEX"

# The open-rl trainer's runtime imports.
uv pip install --python "$V" \
  packaging ninja pybind11 einops peft accelerate safetensors rich regex pyyaml tqdm omegaconf \
  "$TRANSFORMERS" datasets "chz>=0.4.0" fastapi pydantic "redis>=5.0.0" uvicorn httpx psutil setuptools \
  opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-gcp-trace

# nemo-automodel at the pinned commit with its dependencies, torch held at the
# cu129 build. cutlass-dsl is floored at 4.6.0: the 4.5.x that vllm 0.25.x
# resolves cannot compile the FA4 CuTe kernels Gemma 4's head_dim selects.
uv pip install --python "$V" "nemo-automodel @ git+https://github.com/NVIDIA-NeMo/Automodel.git@$AUTOMODEL_SHA" "torch==2.11.0" "$TRANSFORMERS" \
  "nvidia-cutlass-dsl>=4.6.0" \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# GDN kernels: fla for the delta rule (its ops.cp is what Automodel's GDN
# context parallelism runs on), tilelang for the Hopper backward. tilelang
# brings its own apache-tvm-ffi pin; 0.1.9 needed it held at 0.1.9 by hand.
uv pip install --python "$V" "flash-linear-attention==0.5.1" "tilelang>=0.1.11" "torch==2.11.0" \
  --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
# PyPI has no causal-conv1d wheel for torch 2.11. Astral's GPU index builds one
# per torch and CUDA version, so the pin carries both.
uv pip install --python "$V" --no-deps "causal-conv1d==1.7.0+cu.12.9.torch.2.11" \
  --index-url https://wheels.astral.sh/simple/cu129/

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$REPO/src" "$V" - <<'EOF'
import os

import torch, transformers
from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM  # noqa: F401
import fla.ops.cp  # noqa: F401
import tilelang  # noqa: F401
import causal_conv1d  # noqa: F401
# The image checks these after copying src, so its venv layer does not depend on the source.
if os.environ.get("AUTOMODEL_CHECK_REPO", "1") == "1":
  import server.training_requests_processor  # noqa: F401
  import training.automodel_worker  # noqa: F401

print(f"automodel env OK: torch {torch.__version__}, transformers {transformers.__version__}")
EOF
echo "[setup] done: $V"
